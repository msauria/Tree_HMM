from multiprocessing.shared_memory import SharedMemory

import numpy


class TransitionMatrix():
    """Base class for holding hmm transition probabilities"""

    def __init__(self, num_states=None, transition_matrix=None, transition_id=None, RNG=numpy.random):
        assert (not num_states is None) != (not transition_matrix is None)
        assert ((transition_id is None) or
                ((transition_matrix is not None) and (transition_id.shape == transition_matrix.shape)) or
                (transition_id.shape == (num_states, num_states)))
        if not num_states is None:
            self.transition_matrix = numpy.zeros((int(num_states), int(num_states)), numpy.float64)
            self.transition_matrix = RNG.random(size=self.transition_matrix.shape)
            self.update_mask = numpy.ones(self.transition_matrix.shape, bool)
        else:
            self.transition_matrix = numpy.copy(transition_matrix).astype(numpy.float64)
        self.num_states = self.transition_matrix.shape[0]
        if transition_id is not None:
            self.transition_id = numpy.copy(transition_id).astype(numpy.int32)
            self.transition_matrix[numpy.where(self.transition_id < 0)] = 0
        else:
            self.transition_id = numpy.arange(self.num_states ** 2).reshape(self.num_states, -1)
        self.num_transitions = numpy.amax(self.transition_id) + 1
        where = numpy.where(self.transition_id < 0)
        self.transition_matrix[where] = 0
        self.transition_matrix /= numpy.sum(self.transition_matrix, axis=1, keepdims=True)
        self.log_transition_matrix = numpy.zeros_like(self.transition_matrix)
        self.log_transition_matrix.fill(-numpy.inf)
        where = numpy.where(self.transition_matrix > 0)
        self.log_transition_matrix[where] = numpy.log(self.transition_matrix[where])
        self.valid_trans = numpy.where(self.transition_id >= 0)
        self.tallies = numpy.zeros(self.valid_trans[0].shape[0], numpy.float64)
        return

    def __getitem__(self, idx):
        return self.log_transition_matrix[idx]

    def __setitem__(self, idx, value):
        self.transition_matrix[idx] = value
        return

    def clear_tallies(self):
        self.tallies[:] = 0
        self.updated = False
        return

    @classmethod
    def update_tallies(self, *args):
        s, e, pairs, node_children, transitions, smm_map = args
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['forward']))
        forward = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((num_seqs, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        tallies = numpy.zeros((num_states, num_states), numpy.float64)
        for idx1, idx2 in pairs:
            children = node_children[idx1]
            if len(children) == 0:
                tmpreverse = probs[s:e, :, idx1] - scale[s:e, idx1].reshape(-1, 1)
            elif len(children) == 1:
                tmpreverse = (probs[s:e, :, idx1] - scale[s:e, idx1].reshape(-1, 1) +
                              reverse[s:e, :, children[0]])
            else:
                tmpreverse = (probs[s:e, :, idx1] - scale[s:e, idx1].reshape(-1, 1) +
                              numpy.sum(reverse[s:e, :, children], axis=2))
            xi = (forward[s:e, :, idx2].reshape(-1, num_states, 1) +
                  tmpreverse.reshape(-1, 1, num_states) +
                  transitions.reshape(1, num_states, num_states))
            xi -= numpy.amax(xi.reshape(e - s, -1), axis=1).reshape(-1, 1, 1)
            xi = numpy.exp(xi)
            xi /= numpy.sum(numpy.sum(xi, axis=2), axis=1).reshape(-1, 1, 1)
            tallies += numpy.sum(xi, axis=0)
        for V in views:
            V.close()
        return tallies.reshape(-1)

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        tallies = numpy.bincount(self.transition_id[self.valid_trans],
                                 weights=self.tallies,
                                 minlength=self.num_transitions)
        tallies /= numpy.maximum(1, numpy.bincount(self.transition_id[self.valid_trans],
                                                   minlength=self.num_transitions))
        self.transition_matrix.fill(0)
        self.transition_matrix[self.valid_trans] = tallies[
            self.transition_id[self.valid_trans]]
        self.transition_matrix /= numpy.sum(self.transition_matrix, axis=1,
                                            keepdims=True)
        where = numpy.where(self.transition_matrix > 0)
        self.log_transition_matrix.fill(-numpy.inf)
        self.log_transition_matrix[where] = numpy.log(self.transition_matrix[where])
        return

    def print(self):
        output = []
        output.append("Transitions")
        tmp = " ".join([f"{x}".rjust(6, ' ') for x in range(self.num_states)])
        output.append(f" State {tmp}")
        for i in range(self.num_states):
            tmp = " ".join([f"{x*100:0.1f}%".rjust(6, ' ') for x in self.transition_matrix[i, :]])
            state = f"{i}".rjust(5, ' ')
            output.append(f" {state} {tmp}")
        output = "\n".join(output)
        return output

    def generate_transition(self, state, RNG):
        return numpy.searchsorted(numpy.cumsum(
            self.transition_matrix[state, :]), RNG.random())











