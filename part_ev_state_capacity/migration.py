"""Explicit, narrowly scoped Slurm migration. No old campaign files are written."""
import getpass
import re
import subprocess
from pathlib import Path
from . import protocol as P

TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}


class Slurm:
    def call(self, *args):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
        return result.stdout.strip()

    def job(self, job):
        result = subprocess.run(['scontrol', 'show', 'job', job, '-o'],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            if 'Invalid job id' not in result.stderr:
                raise RuntimeError(result.stderr.strip())
            return None
        # Values such as UserId and Command are kept verbatim; no shell evaluation.
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1:
            raise ValueError(f'Ambiguous Slurm record for {job}')
        return dict(re.findall(r'(\w+)=(.*?)(?=\s+\w+=|$)', rows[0]))

    def terminal_state(self, job):
        rows = self.call('sacct', '-n', '-X', '-P', '-j', job, '-o', 'JobID,State').splitlines()
        parent, _, index = job.partition('_')
        states = []
        for line in rows:
            fields = line.strip().split('|')
            if len(fields) < 2:
                continue
            identity, state = fields[:2]
            compressed = (index.isdigit() and identity.startswith(parent + '_[')
                          and identity.endswith(']')
                          and includes_index(identity[len(parent)+1:], int(index)))
            if identity == job or compressed:
                states.append(state.split()[0].rstrip('+'))
        if len(states) != 1 or states[0] not in TERMINAL:
            raise ValueError(f'Cannot establish terminal state of {job}; inspect sacct manually')
        return states[0]


def includes_index(value, index):
    """Interpret Slurm's bounded task list; never use a range to cancel training."""
    values = value.strip('[]').split('%')[0]
    for term in values.split(','):
        if re.fullmatch(r'\d+', term) and int(term) == index:
            return True
        if re.fullmatch(r'\d+-\d+', term):
            lo, hi = map(int, term.split('-'))
            if 0 <= lo <= index <= hi <= 6:
                return True
    return False


def verified_job(slurm, contract, job, phase, index=None, *, allow_compressed=False):
    row = slurm.job(job)
    if row is None:
        return None
    expected_script = str(Path(contract['path']) / 'source/sbatch/run_part_ev_state_fullscale.sh')
    if (row.get('UserId', '').split('(')[0] != getpass.getuser()
            or row.get('Command') != expected_script
            or row.get('JobName') != f'evfull_{phase}'):
        raise ValueError(f'Owner, script or name differs for {job}; nothing cancelled')
    parent = contract['jobs'][phase]
    if index is not None:
        valid = (includes_index(row.get('ArrayTaskId', ''), index) if allow_compressed
                 else row.get('ArrayTaskId') == str(index))
        if row.get('ArrayJobId') != parent or not valid:
            raise ValueError(f'{job} is not an individual expected array task; refusing cancellation')
    elif phase == 'evaluate':
        if row.get('ArrayJobId', row.get('JobId')) != parent:
            raise ValueError('Wrong evaluation array')
    elif row.get('JobId') != parent:
        raise ValueError('Wrong report job')
    return row


def kept_dependencies(root, source, contract, slurm):
    from part_ev_state_fullscale.reporting import validate_training
    dependencies = []
    for index in P.CONTROL_INDICES:
        job = f"{contract['jobs']['train']}_{index}"
        row = verified_job(slurm, contract, job, 'train', index, allow_compressed=True)
        state = row['JobState'] if row else slurm.terminal_state(job)
        if state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
            dependencies.append(job)
        elif state == 'COMPLETED':
            validate_training(root, source, source['tasks'][index], contract['runtime'])
        else:
            raise ValueError(f'Kept control {job} is {state}; recover it before submitting replacements')
    return dependencies


def replacement_status(root, source, contract, slurm, *, apply=False, require_switched=False):
    """Only cancel pending last-four runs; never cancel a running training implicitly."""
    kept = kept_dependencies(root, source, contract, slurm)
    targets = []
    for index in (3, 4, 5):
        job = f"{contract['jobs']['train']}_{index}"
        # show job <array>_<index> can return a compressed array. Read it first,
        # but require a materialized singleton before any cancellation.
        row = slurm.job(job)
        if row:
            expected = str(Path(contract['path']) / 'source/sbatch/run_part_ev_state_fullscale.sh')
            if (row.get('UserId', '').split('(')[0] != getpass.getuser()
                    or row.get('Command') != expected or row.get('JobName') != 'evfull_train'
                    or row.get('ArrayJobId') != contract['jobs']['train']
                    or not includes_index(row.get('ArrayTaskId', ''), index)):
                raise ValueError(f'Unexpected training record for {job}')
            state = row['JobState']
        else:
            state = slurm.terminal_state(job)
        if state not in ('PENDING', 'CANCELLED'):
            raise ValueError(f'{job} is {state}, not pending/cancelled. Inspect before changing running work.')
        targets.append((job, index, state))
    downstream = []
    for phase in ('evaluate', 'report'):
        job = contract['jobs'][phase]
        row = verified_job(slurm, contract, job, phase)
        if row is None and phase == 'evaluate':
            # Parent arrays may lack an accounting row; check every expected element.
            states = [slurm.terminal_state(f'{job}_{i}') for i in range(7)]
            state = 'CANCELLED' if all(s in TERMINAL for s in states) else 'UNKNOWN'
        else:
            state = row['JobState'] if row else slurm.terminal_state(job)
        if state not in TERMINAL and state != 'PENDING':
            raise ValueError(f'Old {phase} {job} is {state}; inspect before cancellation')
        downstream.append((phase, job, state))
    if require_switched and (any(s != 'CANCELLED' for _, _, s in targets)
                             or any(s not in TERMINAL for _, _, s in downstream)):
        raise ValueError('Apply the explicit switch first; old last-four/downstream jobs remain queued')
    plan = dict(kept_dependencies=kept, replace=targets, supersede=downstream,
                applied=False, old_campaign_files_modified=False)
    if apply:
        # Validate the ENTIRE plan before the first mutation. Hold each target to
        # stop scheduling races, then require a singleton, non-parent numeric ID.
        for job, index, state in targets:
            if state == 'CANCELLED':
                continue
            slurm.call('scontrol', 'hold', job)
            row = verified_job(slurm, contract, job, 'train', index)
            numeric = (row or {}).get('JobId', '')
            if (not row or row['JobState'] != 'PENDING' or not numeric.isdigit()
                    or numeric == contract['jobs']['train']):
                raise ValueError(f'Cannot safely isolate {job}; target may be held. No broad cancel attempted.')
            slurm.call('scancel', numeric)
        for phase, job, state in downstream:
            if state in TERMINAL:
                continue
            row = verified_job(slurm, contract, job, phase)
            if row is None:
                checks = [f'{job}_{i}' for i in range(7)] if phase == 'evaluate' else [job]
                for check in checks:
                    slurm.terminal_state(check)
                continue
            if row['JobState'] in TERMINAL:
                continue  # Slurm may already have cancelled the failed dependency chain.
            if row['JobState'] != 'PENDING':
                raise ValueError(f'{job} changed state during switch; inspect before retrying')
            # Intentionally cancel the entire *evaluation* array, never training.
            slurm.call('scancel', job)
        plan['applied'] = True
    return plan
