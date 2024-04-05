import sys
from multiprocessing.shared_memory import SharedMemory

import numpy

from .emission import *
from .state import State
from .transition import TransitionMatrix

class Node():
    """Base class for tree hmm model node"""

    def __init__(self, label, index=0):
        self.name = 'Node'
        self.label = label
        self.parent = None
        self.parent_name = None
        self.children = []
        self.children_names = []
        self.index = index
        self.level = None
        self.root = False
        self.leaf = False
        self.features = None
        self.views = {}
        return

    def find_levels(self, level=0):
        if self.level is None:
            self.level = level
        else:
            raise RuntimeError(f"Node {self.name} appears in tree in multiple positions")
        if level == 0:
            self.root = True
        level += 1
        for n in self.children:
            n.find_levels(level)
        if len(self.children) == 0:
            self.leaf = True
        return

    def print_tree(self):
        if self.level is None:
            level = 0
        else:
            level = self.level
        output = f"{level * '  '}{self.label}\n"
        for c in self.children:
            output += c.print_tree()
        return output

    def find_pairs(self):
        pairs = {}
        if self.parent is not None:
            pairs[self.name] = self.parent.name
        for c in self.children:
            pairs.update(c.find_pairs())
        return pairs

    def initialize_node(self,
                       state_names=None,
                       emissions=None,
                       seed=None):
        self.RNG = numpy.random.default_rng(seed)
        self.num_states = len(emissions)
        self.num_dists = 0
        self.states = []
        for h, E in enumerate(emissions):
            assert (issubclass(type(E), EmissionDistribution) or
                    type(E) == list)
            if issubclass(type(E), EmissionDistribution):
                E = [E]
            self.num_emissions = len(E)
            self.states.append(State(E, h))
            if state_names is not None and h < len(state_names):
                self.states[-1].label = state_names[h]
        return

    def find_levels(self, level=0):
        if self.level is None:
            self.level = level
        else:
            raise RuntimeError(f"Node {self.name} appears in tree in multiple positions")
        if level == 0:
            self.root = True
        level += 1
        for n in self.children:
            n.find_levels(level)
        if len(self.children) == 0:
            self.leaf = True
        return

    def get_emission_dtype(self):
        dtypes = []
        for i in range(self.states[0].num_distributions):
            if issubclass(type(self.states[0].distributions[i]), EmissionContinuousDistribution):
                dtypes.append(numpy.float64)
            else:
                dtypes.append(numpy.int32)
        return numpy.dtype([(f"{x}", dtypes[x]) for x in range(len(dtypes))])

    @classmethod
    def generate_sequences(cls, *args):
        start, end, root, initprobs, transitions, smm_map, seed = args
        obs_dtype = root.states[0].get_observation_dtype()
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        RNG = numpy.random.default_rng(seed)
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((num_seqs, num_nodes), obs_dtype, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_states']))
        states = numpy.ndarray((num_seqs, num_nodes,), numpy.int32, buffer=views[-1].buf)
        initprobs = numpy.cumsum(initprobs)
        for i in range(start, end):
            states[i, root.index] = numpy.searchsorted(initprobs, RNG.random())
            root.generate_sequence(states[i, :], obs[i, :], transitions, RNG)
        for V in views:
            V.close()
        return

    def generate_sequence(self, states, obs, transitions, RNG):
        state = states[self.index]        
        obs[self.index] = self.states[state].generate_sequence(RNG)
        for C in self.children:
            states[C.index] = transitions.generate_transition(state, RNG)
            C.generate_sequence(states, obs, transitions, RNG)
        return

    def clear_tallies(self):
        for S in self.states:
            S.clear_tallies()
        return

    def get_parameters(self):
        params = []
        for i in range(self.num_states):
            params.append(self.states[i].get_parameters())
        return params

    @classmethod
    def find_paths(self, *args):
        (start, end, initprobs, transitions, node_parents, node_children,
            node_order, smm_map) = args
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_states']))
        states = numpy.ndarray((num_seqs, num_nodes), dtype=numpy.int32,
                               buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_scores']))
        scores = numpy.ndarray((num_seqs, 2), dtype=numpy.float64,
                               buffer=views[-1].buf)
        root_idx = node_order[0]
        for i in range(start, end):
            path = numpy.zeros((num_states, num_nodes), numpy.int32)
            all_scores = numpy.zeros((num_states, num_states, num_nodes), numpy.float64)
            for j in node_order[::-1]:
                children_idxs = node_children[j]
                reverse[i, :, j] = numpy.copy(probs[i, :, j])
                for idx in children_idxs:
                    best = numpy.argmax(all_scores[:, :, idx], axis=1)
                    path[:, idx] = best
                    reverse[i, :, j] += all_scores[numpy.arange(num_states), best, idx]
                if j != node_order[0]:
                    all_scores[:, :, j] = reverse[i:i+1, :, j] + transitions
                else:
                    reverse[i, :, j] += initprobs
                    all_scores[:, :, j] = reverse[i:i+1, :, j]
            best = numpy.argmax(reverse[i, :, root_idx])
            states[i, root_idx] = best
            scores[i, 0] = reverse[i, best, root_idx]
            for j in node_order[1:]:
                states[i, j] = path[states[i, node_parents[j]], j]
        for V in views:
            V.close()
        return

    @classmethod
    def find_probs(self, *args):
        start, end, obsDtype, node, smm_map = args
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        node_idx = node.index
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((num_seqs, num_nodes), obsDtype, buffer=views[-1].buf)
        if 'obs_mask' in smm_map:
            views.append(SharedMemory(smm_map['obs_mask']))
            obs_mask = numpy.ndarray((num_seqs, num_states), bool,
                                     buffer=views[-1].buf)
        else:
            obs_mask = None
        marks = obs.dtype.names
        kwargs = {'start': start, "end": end, "smm_map": smm_map}
        views.append(SharedMemory(smm_map['dist_probs']))
        dist_probs = numpy.ndarray((num_seqs, num_dists, num_nodes),
                                   numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        updated = numpy.zeros(num_dists, bool)
        state_updated = {}
        for i in range(num_states):
            indices = []
            for j, D in enumerate(node.states[i].distributions):
                indices.append(D.index)
                if updated[D.index]:
                    continue
                if D.updated and D.fixed:
                    continue
                dist_probs[start:end, D.index, node_idx] = D.score_observations(
                    obs[marks[j]][start:end, node_idx], **kwargs)
                updated[D.index] = True
            indices = tuple(indices)
            if indices in state_updated:
                probs[start:end, i, node_idx] = numpy.copy(
                    probs[start:end, state_updated[indices], node_idx])
            else:
                probs[start:end, i, node_idx] = numpy.sum(
                    dist_probs[start:end, indices, node_idx], axis=1)
                state_updated[indices] = i
        if obs_mask is not None:
            where = numpy.where(numpy.logical_not(obs_mask[start:end, :]))
            probs[where[0] + start, where[1], node_idx] = -numpy.inf
        for V in views:
            V.close()
        return

    @classmethod
    def find_reverse(self, *args):
        start, end, transitions, node_children, node_order, smm_map = args
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((num_seqs, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        reverse[start:end, :, node_order[0]] = 0
        for i in node_order[1:][::-1]:
            children_idxs = node_children[i]
            if len(children_idxs) > 0:
                reverse[start:end, :, i] = (numpy.sum(
                    reverse[start:end, :, children_idxs], axis=2) +
                    probs[start:end, :, i])
            else:
                reverse[start:end, :, i] = probs[start:end, :, i]
            reverse[start:end, :, i] = scipy.special.logsumexp(
                reverse[start:end, :, i].reshape(-1, 1, num_states) + 
                transitions.reshape(1, num_states, num_states), axis=2)
            scale[start:end, i] = scipy.special.logsumexp(
                reverse[start:end, :, i], axis=1)
            reverse[start:end, :, i] -= scale[start:end, i].reshape(-1, 1)
        for V in views:
            V.close()
        return

    @classmethod
    def find_forward(self, *args):
        (start, end, initprobs, transitions, node_parents, node_children,
         node_order, smm_map) = args
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
        for i in node_order:
            parent_idx = node_parents[i]
            if parent_idx is None:
                forward[start:end, :, i] = (initprobs.reshape(1, -1) +
                                             probs[start:end, :, i])
                scale[start:end, i] = scipy.special.logsumexp(
                    forward[start:end, :, i], axis=1)
                forward[start:end, :, i] -= scale[start:end, i].reshape(-1, 1)
            else:
                children_idxs = node_children[parent_idx]
                if children_idxs.shape[0] == 1:
                    forward[start:end, :, i] = (scipy.special.logsumexp(
                        forward[start:end, :, parent_idx].reshape(-1, num_states, 1) +
                        transitions.reshape(1, num_states, num_states), axis=1) +
                        probs[start:end, :, i] - scale[start:end, i].reshape(-1, 1))
                else:
                    children_idxs = children_idxs[numpy.where(children_idxs != i)[0]]
                    tmp = forward[start:end, :, parent_idx] + numpy.sum(
                        reverse[start:end, :, children_idxs], axis=2)
                    forward[start:end, :, i] = (scipy.special.logsumexp(
                        tmp.reshape(-1, num_states, 1) +
                        transitions.reshape(1, num_states, num_states), axis=1) +
                        probs[start:end, :, i] - scale[start:end, i].reshape(-1, 1))
        for V in views:
            V.close()
        return

    @classmethod
    def find_total(self, *args):
        s, e, node_children, node_order, smm_map = args
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['forward']))
        forward = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['log_total']))
        log_total = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                                  buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['total']))
        total = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((num_seqs, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_scores']))
        scores = numpy.ndarray((num_seqs, 2),
                               dtype=numpy.float64, buffer=views[-1].buf)
        for i in node_order:
            children = node_children[i]
            if children.shape[0] == 0:
                log_total[s:e, :, i] = forward[s:e, :, i]
            else:
                log_total[s:e, :, i] = forward[s:e, :, i] + numpy.sum(
                    reverse[s:e, :, children], axis=2)

        total[s:e, :, :] = numpy.exp(log_total[s:e, :, :] - numpy.amax(
            log_total[s:e, :, :], axis=1, keepdims=True))
        total[s:e, :, :] /= numpy.sum(total[s:e, :, :], axis=1,
                                             keepdims=True)
        root = node_order[0]
        tmp = scipy.special.logsumexp(log_total[s:e, :, root], axis=1)
        scores[s:e, 0] = numpy.sum(scale[s:e, :], axis=1)
        scores[s:e, 1] = tmp + scores[s:e, 0]
        for V in views:
            V.close()
        return

    def __str__(self):
        dists = {}
        for S in self.states:
            for D in S.distributions:
                dists[D.index] = D
        dists = list(dists.values())
        output = []
        output.append(f"{self.label} node")
        output.append(f"Distributions")
        just = max([len(x.label) for x in dists])
        for D in dists:
            tmp = [D.label.rjust(just)] + [f"{name}:{value}" for name, value in
                               D.get_parameters(log=False).items()]
            output.append(f'  {" ".join(tmp)}')
        just2 = max([len(x.label) for x in self.states])
        output.append("\nStates")
        for S in self.states:
            tmp = [S.label.rjust(just2)] + [", ".join([x.label.rjust(just) for x in S.distributions])]
            output.append(f'  {" ".join(tmp)}')
        output = "\n".join(output)
        return output

    def print_distributions(self):
        for D in self.distributions:
            tmp = [D.label] + [f"{name}:{value}" for name, value in
                               D.get_parameters().items()]
            print(" ".join(tmp))
        return

    @classmethod
    def product(cls, X):
        prod = 1
        for x in X:
            prod *= x
        return prod















