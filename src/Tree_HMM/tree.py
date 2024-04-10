import sys
import multiprocessing.managers
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import time
import copy

import numpy
import scipy
import matplotlib.pyplot as plt

from .emission import *
from .state import State
from .transition import TransitionMatrix
from .node import Node
from .initial_probability import InitialProbability

class Tree():
    """Base class for hmm model"""

    def __init__(self,
                 tree=None,
                 emissions=None,
                 state_names=None,
                 transition_matrix=None,
                 transition_id=None,
                 initial_probabilities=None,
                 initial_probability_id=None,
                 RNG=None,
                 nthreads=1,
                 fname=None):
        self.name = "Tree"
        if RNG is None:
            self.RNG = numpy.random.default_rng()
        else:
            self.RNG = RNG
        self.smm_map = None
        self.views = None
        self.num_obs = None
        self.nthreads = max(1, int(nthreads))
        if fname is not None:
            self.load(fname)
            return
        if tree is None or emissions is None:
            raise ValueError("tree and emissions must be passed if model not " \
                             "being loaded from a file")
        self.load_tree(tree)
        assert isinstance(emissions, list) or isinstance(emissions, dict)
        if isinstance(emissions[0], list):
            if isinstance(emissions[0][0], list):
                self.num_states = len(emissions[0])
                assert len(emissions) == self.num_nodes
                node_emissions = emissions
            else:
                assert issubclass(type(emissions[0][0]), EmissionDistribution)
                self.num_states = len(emissions)
                node_emissions = [copy.deepcopy(emissions) for x in range(self.num_nodes)]
        else:
            node_emissions = [emissions[x] for x in self.node_order]
            self.num_states = len(node_emissions[0])
        if transition_matrix is not None:
            node_transitions = TransitionMatrix(transition_matrix=transition_matrix,
                                                transition_id=transition_id)
        else:
            node_transitions = TransitionMatrix(num_states=self.num_states,
                                                RNG=self.RNG)
        if initial_probabilities is not None:
            init_probs = InitialProbability(initial_probabilities=initial_probabilities,
                                            initial_mapping=initial_probability_id,
                                            seed=self.RNG.integers(0, 100000000))
        else:
            init_probs = InitialProbability(num_states = self.num_states,
                                            seed=self.RNG.integers(0, 100000000))
        self.initial_probabilities = init_probs
        for i, name in enumerate(self.node_order):
            node = self.nodes[name]
            node.initialize_node(state_names=state_names, emissions=node_emissions[i])
        self.num_dists = 0
        for name in self.node_order:
            N = self.nodes[name]
            for S in N.states:
                for D in S.distributions:
                    if D.index is None:
                        D.index = self.num_dists
                        self.num_dists += 1
                        if D.name.endswith("Mixture"):
                            for i, D1 in enumerate(D.distributions):
                                if D1.index is None:
                                    D1.index = self.num_dists
                                    self.num_dists += 1
                                D.distribution_indices[i] = D1.index
        self.transitions = node_transitions
        return

    def __str__(self):
        output = []
        output.append(f"Tree model -")
        output.append(self.root.print_tree())
        output.append(f"\nInitial Probabilities")
        tmp = [f"{x:0.3f}" for x in self.initial_probabilities.initial_probabilities]
        output.append(f'  {", ".join(tmp)}')
        output.append("")
        output.append(self.transitions.print())
        output.append("")
        for name in self.node_order:
            output.append(self.nodes[name].__str__())
            output.append('')
        return "\n".join(output)

    @classmethod
    def product(cls, X):
        prod = 1
        for x in X:
            prod *= x
        return prod

    def load_tree(self, tree):
        # Evaluate tree
        if type(tree) == str:
            parents = eval(open(tree).read().strip())
        elif type(tree) != dict:
            raise RuntimeError("Incorrect tree format")
        else:
            parents = tree
        # Add nodes and connections
        self.nodes = {}
        self.node_index = {}
        self.root = None
        node_order = []
        for c, p in parents.items():
            node_order.append(c)
            if p not in self.nodes and p is not None:
                self.node_index[p] = len(self.nodes)
                self.nodes[p] = Node(p, self.node_index[p])
            if c not in self.nodes:
                self.node_index[c] = len(self.nodes)
                self.nodes[c] = Node(c, self.node_index[c])
            if p is not None:
                self.nodes[c].parent = self.nodes[p]
                self.nodes[c].parent_name = p
                self.nodes[p].children.append(self.nodes[c])
                self.nodes[p].children_names.append(c)
        # Verify that there is a single root
        for node in self.nodes.values():
            if node.parent is None:
                if self.root is None:
                    self.root = node
                else:
                    raise RuntimeError(f"More than one root found in tree [{self.root},{node.name}]")
        # Label levels in tree nodes
        self.root.find_levels()
        for n in self.nodes.values():
            if n.level is None:
                raise RuntimeError("Not all nodes are in a single tree")
        # Figure out order to evaluate nodes in
        levels = {}
        for name in node_order:
            n = self.nodes[name]
            l = n.level
            levels.setdefault(l, [])
            levels[l].append(name)
        L = list(levels.keys())
        L.sort()
        order = []
        for l in L:
            order += levels[l]
        self.node_order = order
        self.num_nodes = len(self.nodes)
        self.node_idx_order = numpy.zeros(len(self.node_order), dtype=numpy.int32)
        self.node_parents = {}
        self.node_children = {}
        self.node_pairs = []
        for i, name in enumerate(self.node_order):
            idx = self.nodes[name].index
            self.node_idx_order[i] = idx
            if self.nodes[name].parent is None:
                self.node_parents[idx] = None
            else:
                self.node_parents[idx] = self.nodes[name].parent.index
                self.node_pairs.append((idx, self.nodes[name].parent.index))
            self.node_children.setdefault(idx, [])
            for child in self.nodes[name].children:
                self.node_children[idx].append(child.index)
        for k, v in self.node_children.items():
                self.node_children[k] = numpy.array(v, numpy.int32)
        self.node_pairs = numpy.array(self.node_pairs)
        return

    def make_shared_array(self, name, shape, dtype, data=None):
        if name in self.views:
            new_size = ((self.product(shape) * numpy.dtype(dtype).itemsize - 1) //
                        4096 + 1) * 4096
            if self.views[name].size != new_size:
                self.views[name].unlink()
                del self.views[name]
                self.views[name] = self.smm.SharedMemory(self.product(shape) *
                                                    numpy.dtype(dtype).itemsize)
        else:
            self.views[name] = self.smm.SharedMemory(self.product(shape) *
                                                numpy.dtype(dtype).itemsize)
        self.smm_map[name] = self.views[name].name
        new_data = numpy.ndarray(shape, dtype, buffer=self.views[name].buf)
        if data is not None:
            new_data[:] = data
        setattr(self, name, new_data)
        return new_data

    def ingest_observations(self, obs, names, obs_mask=None):
        assert self.smm_map is not None, "Tree must be used inside a 'with' statement"
        if type(obs) != list:
            obs = [obs]
        for O in obs:
            assert type(O) == numpy.ndarray and O.dtype.names is not None
        self.num_seqs = len(obs)
        self.make_shared_array(f"obs", (self.num_seqs, self.num_nodes), obs[0].dtype)
        self.make_shared_array(f"scale", (self.num_seqs, self.num_nodes),
                               numpy.float64)
        indices = numpy.zeros(self.num_nodes, numpy.int32)
        for node in self.nodes.values():
            indices[node.index] = names.index(node.label)
        for i in range(len(obs)):
            self.obs[i, :] = obs[i][indices]
        self.thread_seq_indices = numpy.round(numpy.linspace(
            0, self.num_seqs, self.nthreads + 1)).astype(numpy.int32)
        self.make_shared_array(f"obs_scores", (self.num_seqs, 2),
                               numpy.float64)
        if obs_mask is not None:
            self.make_shared_array(f"obs_mask", (self.num_seqs, self.num_states), bool)
            for i in range(self.num_seqs):
                self.obs_mask[i, :, :] = obs_mask[i][indices, :].astype(bool)
        self.make_shared_array(f"sizes", (4,), numpy.int64)
        self.sizes[:] = (self.num_nodes, self.num_dists,  self.num_states,
                         self.num_seqs)
        return


    def train_model(self, obs=None, obs_names=None, obs_mask=None, iterations=100, node_burnin=5, tree_burnin=2, min_delta=1e-8):
        if obs is not None:
            self.ingest_observations(obs, obs_names, obs_mask)
        else:
            assert self.num_obs is not None
        self.make_shared_array(f"dist_probs",
                               (self.num_seqs, self.num_dists),
                               numpy.float64)
        self.make_shared_array(f"probs",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        self.make_shared_array(f"forward",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        self.make_shared_array(f"reverse",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        self.make_shared_array(f"log_total",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        self.make_shared_array(f"total",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        for i, name in enumerate(self.obs.dtype.names):
            for N in self.nodes.values():
                idx = N.index
                if numpy.amax(self.obs[name][:, idx]) == numpy.amin(self.obs[name][:, idx]):
                    for S in N.states:
                        S.distributions[i].fixed = True
                        S.distributions[i].updated = True
                        self.dist_probs[:, S.distributions[i].index, idx] = 1
        prev_prob = 1
        for i in range(iterations):
            prob = self.update_model()
            print(f"\r{" "*80}\rIteration {i}: Log-prob {prob: 0.1f}", end='\n',
                  file=sys.stderr)
            #self.plot_params(i)
            # self.save(f"model_iter{i}.npz")
            if abs(prob - prev_prob) / abs(prev_prob) < min_delta:
                break
            prev_prob = prob
        print(f"\r{' ' * 80}\r", end="", file=sys.stderr)
        return

    def update_model(self):
        self.clear_tallies()
        self.find_probs()
        self.find_reverse()
        self.find_forward()
        self.find_total()
        self.update_tallies()
        self.apply_tallies()
        return -self.log_probs

    def generate_sequences(self, num_seqs=1):
        assert self.smm_map is not None, "HmmManager must be used inside a 'with' statement"
        self.num_seqs = int(num_seqs)
        self.make_shared_array(f"sizes", (4,), numpy.int64)
        self.sizes[:] = (self.num_nodes, self.num_dists, self.num_states,
                         self.num_seqs)
        self.make_shared_array(f"obs", (self.num_seqs, self.num_nodes),
                               self.root.get_emission_dtype())
        self.make_shared_array(f"obs_states", (self.num_seqs, self.num_nodes),
                               numpy.int32)
        self.thread_seq_indices = numpy.round(numpy.linspace(
            0, self.num_seqs, self.nthreads + 1)).astype(numpy.int32)
        kwargs = {
            'initprobs': self.initial_probabilities.initprobsum,
            'transitions': self.transitions,
            'obsDtype': self.root.get_emission_dtype(),
            'root': self.root,
            'seed': self.RNG.integers(0, 99999999),
            'smm_map': self.smm_map
        }
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            kwargs['start'] = s
            kwargs['end'] = e
            futures.append(self.pool.apply_async(Node.generate_sequences,
                                                 kwds=copy.deepcopy(kwargs)))
        for result in futures:
            _ = result.get()
        names = ["" for x in range(self.num_nodes)]
        for node in self.nodes.values():
            names[node.index] = node.label
        states = []
        obs = []
        for i in range(self.num_seqs):
            states.append(numpy.copy(self.obs_states[i, :]))
            obs.append(numpy.copy(self.obs[i, :]))
        return obs, states, names

    def viterbi(self, obs, names):
        if type(obs) == numpy.ndarray:
            obs = [obs]
        self.ingest_observations(obs, names)
        self.make_shared_array(f"dist_probs",
                               (self.num_seqs, self.num_dists),
                               numpy.float64)
        self.make_shared_array(f"probs",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        self.make_shared_array(f"reverse",
                               (self.num_seqs, self.num_states, self.num_nodes),
                               numpy.float64)
        self.make_shared_array(f"obs_states", (self.num_seqs, self.num_nodes),
                               numpy.int32)
        self.make_shared_array(f"obs_scores", (self.num_seqs, 2),
                               numpy.float64)
        for i, name in enumerate(self.obs.dtype.names):
            for N in self.nodes.values():
                idx = N.index
                if numpy.amax(self.obs[name][:, idx]) == numpy.amin(self.obs[name][:, idx]):
                    for S in N.states:
                        S.distributions[i].fixed = True
                        S.distributions[i].updated = True
                        self.dist_probs[:, S.distributions[i].index, idx] = 1
        self.find_probs()
        self.find_paths()
        states = []
        for i in range(self.num_seqs):
            states.append(self.obs_states[i, :])
        return states, list(self.obs_scores[:, 0])

    def __enter__(self):
        self.smm_map = {}
        self.views = {}
        self.smm = multiprocessing.managers.SharedMemoryManager()
        self.smm.start()
        self.pool = multiprocessing.Pool(self.nthreads)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()
        self.pool.terminate()
        self.smm.shutdown()
        return

    def find_paths(self):
        kwargs = {
            'initprobs': self.initial_probabilities[:],
            'transitions': self.transitions[:, :],
            'node_parents': self.node_parents,
            'node_children': self.node_children,
            'node_order': self.node_idx_order,
            'smm_map': self.smm_map
        }
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            kwargs['start'] = s
            kwargs['end'] = e
            futures.append(self.pool.apply_async(Node.find_paths,
                                                 kwds=copy.deepcopy(kwargs)))
        for result in futures:
            _ = result.get()
        return

    def clear_tallies(self):
        for node in self.nodes.values():
            node.clear_tallies()
        self.initial_probabilities.clear_tallies()
        self.transitions.clear_tallies()
        return

    def find_probs(self):
        kwargs = {
            'obsDtype': self.obs.dtype,
            'smm_map': self.smm_map
        }
        futures = []
        for node in self.nodes.values():
            kwargs['node'] = node
            for i in range(self.thread_seq_indices.shape[0] - 1):
                s, e = self.thread_seq_indices[i:i+2]
                kwargs['start'] = s
                kwargs['end'] = e
                futures.append(self.pool.apply_async(Node.find_probs,
                                                     kwds=copy.deepcopy(kwargs)))
        for result in futures:
            _ = result.get()
        return

    def find_reverse(self):
        kwargs = {
            'transitions': self.transitions[:, :],
            'node_children': self.node_children,
            'node_order': self.node_idx_order,
            'smm_map': self.smm_map
        }
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            kwargs['start'] = s
            kwargs['end'] = e
            futures.append(self.pool.apply_async(Node.find_reverse,
                                                 kwds=copy.deepcopy(kwargs)))
        for result in futures:
            _ = result.get()
        return

    def find_forward(self):
        kwargs = {
            'initprobs': self.initial_probabilities[:],
            'transitions': self.transitions[:, :],
            'node_parents': self.node_parents,
            'node_children': self.node_children,
            'node_order': self.node_idx_order,
            'smm_map': self.smm_map
        }
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            kwargs['start'] = s
            kwargs['end'] = e
            futures.append(self.pool.apply_async(Node.find_forward,
                                                 kwds=copy.deepcopy(kwargs)))
        for result in futures:
            _ = result.get()
        return

    def find_total(self):
        kwargs = {
            'node_children': self.node_children,
            'node_order': self.node_idx_order,
            'smm_map': self.smm_map
        }
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            kwargs['start'] = s
            kwargs['end'] = e
            futures.append(self.pool.apply_async(Node.find_total,
                                                 kwds=copy.deepcopy(kwargs)))
        for result in futures:
            _ = result.get()
        return

    def update_tallies(self):
        self.log_probs = numpy.sum(self.obs_scores[:, 1])
        self.initial_probabilities.update_tallies(self.total[:, :, self.root.index])
        self.update_transition_tallies()
        self.update_emission_tallies()
        return

    def update_transition_tallies(self):
        kwargs = {
            'node_pairs': self.node_pairs,
            'node_children': self.node_children,
            'transitions': self.transitions[:, :],
            'smm_map': self.smm_map
        }
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            kwargs['start'] = s
            kwargs['end'] = e
            futures.append(self.pool.apply_async(TransitionMatrix.find_tallies,
                                                 kwds=copy.deepcopy(kwargs)))
        for result in futures:
            self.transitions.update_tallies(result.get())
        return

    def update_emission_tallies(self):
        kwargs = {
            'obsDtype': self.obs.dtype,
            'smm_map': self.smm_map
        }
        futures = []
        for node in self.nodes.values():
            kwargs['node_name'] = node.label
            kwargs['node_idx'] = node.index
            for i in range(self.num_states):
                kwargs['state_idx'] = i
                for j, D in enumerate(node.states[i].distributions):
                    if D.fixed:
                        continue
                    kwargs['dist_idx'] = j
                    kwargs['params'] = D.get_parameters()
                    kwargs['func'] = D.find_tallies
                    for k in range(self.thread_seq_indices.shape[0] - 1):
                        s, e = self.thread_seq_indices[k:k+2]
                        kwargs['start'] = s
                        kwargs['end'] = e
                        futures.append(self.pool.apply_async(
                            EmissionDistribution.find_tallies,
                            kwds=copy.deepcopy(kwargs)))
        for result in futures:
            node_name, state_idx, dist_idx, tallies = result.get()
            self.nodes[node_name].states[state_idx].distributions[dist_idx].update_tallies(tallies)
        return

    def apply_tallies(self):
        for N in self.nodes.values():
            for S in N.states:
                for D in S.distributions:
                    D.apply_tallies()
        self.transitions.apply_tallies()
        self.initial_probabilities.apply_tallies()
        return

    def plot_params(self, iteration):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        N = self.num_states
        M = len([None for x in self.obs.dtype.names if x.count('TSS') == 0])
        hm = numpy.zeros((N, M), dtype=numpy.float64)
        for i in range(N):
            for j in range(M):
                hm[i, j] = self.HMM.states[i].distributions[j].distributions[0].mu
        hm /= numpy.amax(hm, axis=0, keepdims=True)
        ax[0].imshow(hm, aspect='auto')
        for i in range(N):
            for j in range(M):
                hm[i, j] = self.HMM.states[i].distributions[j].distributions[0].sigma
        hm /= numpy.amax(hm, axis=0, keepdims=True)
        ax[1].imshow(hm, aspect='auto')
        plt.savefig(f"param_{iteration}.pdf")
        return

    def save(self, fname):
        data = {}
        data['transition_matrix'] = self.transitions.transition_matrix
        data['transition_id'] = self.transitions.transition_id
        data['initial_probabilities'] = self.initial_probabilities.initial_probabilities
        data['initial_mapping'] = self.initial_probabilities.initial_mapping
        dist_map = []
        for N in self.nodes.values():
            for i in range(self.num_states):
                for j, D in enumerate(N.states[i].distributions):
                    dist_map.append((i, j, N.index, D.index))
                    if f"dist_{D.index}" not in data:
                        data[f"dist_{D.index}"] = self.dist_to_array(D)
                    if D.name.endswith('Mixture'):
                        for k, D1 in enumerate(D.distributions):
                            if f"dist_{D1.index}" not in data:
                                data[f"dist_{D1.index}"] = self.dist_to_array(D1)
        dist_map = numpy.array(dist_map,
            numpy.dtype([('state', numpy.int32), ('dist', numpy.int32),
                         ('node', numpy.int32), ('index', numpy.int32)]))
        data['distribution_mapping'] = dist_map
        state_names = [self.root.states[x].label for x in range(self.num_states)]
        data['state_names'] = numpy.array(
            state_names, f"<U{max([len(x) for x in state_names])}")
        node_names = ["" for x in range(self.num_nodes)]
        for nodename in self.node_order:
            node = self.nodes[nodename]
            node_names[node.index] = node.label
        data['node_names'] = numpy.array(
            node_names, f"<U{max([len(x) for x in node_names])}")
        pairs = [(self.root.label, '')] + [(child, self.nodes[child].parent_name)
                                           for child in self.node_order[1:]]
        maxlen = max([len(x[0]) for x in pairs] + [len(x[1]) for x in pairs])
        data['tree'] = numpy.array(pairs, f"<U{maxlen}")
        numpy.savez(fname, **data)
        return

    def dist_to_array(self, dist):
        params = dist.get_parameters()
        dtype = [('name', f'<U{len(dist.name)}'),
                 ('label', f'<U{len(dist.label)}'),
                 ('fixed', bool)]
        for name, value in params.items():
            if isinstance(value, list):
                continue
            if (isinstance(value, numpy.ndarray) and 
                (len(value.shape) > 1 or value.shape[0] > 1)):
                dtype.append((name, numpy.float64, value.shape))
            else:
                dtype.append((name, numpy.float64))
        data = numpy.zeros(1, dtype=numpy.dtype(dtype))
        data['name'] = dist.name
        data['label'] = dist.label
        data['fixed'] = dist.fixed
        for name, value in params.items():
            if isinstance(value, list):
                continue
            data[name] = value
        return data

    def load(self, fname):
        temp = numpy.load(fname)
        transition_matrix = temp['transition_matrix']
        transition_id = temp['transition_id']
        initial_probabilities = temp['initial_probabilities']
        initial_mapping = temp['initial_mapping']
        state_names = temp['state_names']
        node_names = temp['node_names']
        tree = {x[0]: x[1] for x in temp['tree']}
        for key in tree.keys():
            if tree[key] == '':
                tree[key] = None
        dist_map = temp['distribution_mapping']
        num_states = len(state_names)
        dists = {
            "Base": EmissionDistribution,
            "Alphabet": EmissionAlphabetDistribution,
            "Poisson": EmissionPoissonDistribution,
            "DiscreteMixture": EmissionDiscreteMixtureDistribution,
            "Gaussian": EmissionGaussianDistribution,
            "LogNormal": EmissionLogNormalDistribution,
            "Gamma": EmissionGammaDistribution,
            "Zero": EmissionZeroDistribution,
            "ContinuousMixture": EmissionContinuousMixtureDistribution,
        }
        num_nodes = len(node_names)
        distributions = {}
        for name in temp.keys():
            if not name.startswith('dist_'):
                continue
            params = {}
            for name2 in temp[name].dtype.names:
                if name2 == 'name':
                    dname = temp[name][name2][0]
                elif isinstance(temp[name][name2], numpy.ndarray):
                    if len(temp[name][name2].shape) == 1:
                        params[name2] = temp[name][name2][0]
                    else:
                        params[name2] = temp[name][name2][0, :]
                else:
                    params[name2] = temp[name][name2]
            distributions[f"{name}"] = dists[dname](**params)
            distributions[f"{name}"].index = int(name.split('_')[-1])
        emissions = [[[None for y in range(numpy.amax(dist_map['dist']) + 1)]
                     for x in range(num_states)] for z in range(num_nodes)]
        for s_idx, d_idx, n_idx, idx in dist_map:
            D = distributions[f"dist_{idx}"]
            emissions[n_idx][s_idx][d_idx] = D
            if D.name.endswith('Mixture'):
                mixdists = [distributions[f"dist_{x}"] for x in D.distribution_indices]
                D.distributions = mixdists
        self.num_dists = len(distributions)
        self.load_tree(tree)
        self.num_states = num_states
        transitions = TransitionMatrix(transition_matrix=transition_matrix,
                                       transition_id=transition_id)
        init_probs = InitialProbability(initial_probabilities=initial_probabilities,
                                        initial_mapping=initial_mapping)
        for i, name in enumerate(self.node_order):
            node = self.nodes[name]
            node.initialize_node(state_names, emissions[i])
        self.initial_probabilities = init_probs
        self.transitions = transitions
        self.log_probs = 0
        return









