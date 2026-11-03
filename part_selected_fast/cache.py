"""Bounded set-associative byte arena. No Python object per cached jet, no disk.

Fixed-size slots trade some packing efficiency for a strict predictable bound.
Four-way FIFO replacement affects performance only. Cache buffers are ordinary
pageable RAM, never saved in state_dict/checkpoints, and never contain autograd.
"""
import numpy as np


class TreeCache:
    SLOT_BYTES = (3*255+4*128+2 + 10*255+3*128)*4
    WAYS = 4

    def __init__(self, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('Cache budget must be a nonnegative byte count')
        # Include every array and fixed array-object overhead conservatively.
        overhead, slot = 4096, self.SLOT_BYTES+32+4+1
        self.sets = max(0, (budget_bytes-overhead)//(self.WAYS*slot+1))
        shape = (self.sets, self.WAYS)
        self.keys = np.zeros((*shape,32), np.uint8)
        self.lengths = np.zeros(shape, np.uint32)
        self.valid = np.zeros(shape, np.bool_)
        self.cursor = np.zeros(self.sets, np.uint8)
        self.payload = np.empty((*shape,self.SLOT_BYTES), np.uint8)
        self.allocated_bytes = sum(x.nbytes for x in (self.keys,self.lengths,self.valid,self.cursor,self.payload))
        self.budget_bytes = budget_bytes
        self.hits = self.misses = self.inserts = self.evictions = 0

    def get(self, key):
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError('SHA256 byte key required')
        if self.sets:
            index = int.from_bytes(key[:8], 'little') % self.sets
            match = self.valid[index] & (self.keys[index] == np.frombuffer(key,np.uint8)).all(1)
            found = np.flatnonzero(match)
            if len(found):
                way = int(found[0]); self.hits += 1
                return self.payload[index,way,:int(self.lengths[index,way])].tobytes()
        self.misses += 1
        return None

    def put(self, key, value):
        if not isinstance(key,bytes) or len(key)!=32 or not isinstance(value,bytes) or len(value)>self.SLOT_BYTES:
            raise ValueError('Only fixed-size tree byte records are cacheable')
        if not self.sets:
            return
        index = int.from_bytes(key[:8],'little') % self.sets
        matches = np.flatnonzero(self.valid[index] & (self.keys[index]==np.frombuffer(key,np.uint8)).all(1))
        way = int(matches[0]) if len(matches) else int(self.cursor[index])
        if not len(matches):
            self.evictions += int(self.valid[index,way])
            self.cursor[index] = (way+1)%self.WAYS
        self.valid[index,way] = False
        self.keys[index,way] = np.frombuffer(key,np.uint8)
        self.payload[index,way,:len(value)] = np.frombuffer(value,np.uint8)
        self.lengths[index,way] = len(value)
        self.valid[index,way] = True
        self.inserts += 1

    def clear(self):
        self.valid.fill(False)
        self.cursor.fill(0)

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, inserts=self.inserts, evictions=self.evictions,
                    allocated_bytes=self.allocated_bytes, budget_bytes=self.budget_bytes,
                    capacity=self.sets*self.WAYS)
