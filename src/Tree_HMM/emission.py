from multiprocessing.shared_memory import SharedMemory

import numpy
import scipy.special
import scipy.stats

class EmissionDistribution():
    """Base class for hmm emission distributions"""

    def __init__(self, label=''):
        self.name = "Base"
        self.fixed = False
        self.label = str(label)
        self.updated = True
        return

    def clear_tallies(self):
        if not self.fixed:
            self.tallies[:] = 0
            self.updated = False
        return

    @classmethod
    def update_tallies(self, **kwargs):
        return kwargs['func'](**kwargs)

    def print(self, level=0):
        return f"{' '*level}Base emission distribution {self.label}"

    def get_parameters(self):
        return {}


class EmissionDiscreteDistribution(EmissionDistribution):
    """Discrete hmm emission distribtion"""

class EmissionAlphabetDistribution(EmissionDiscreteDistribution):
    """Alphabet-based hmm emission distribution"""

    def __init__(self, probabilities=None, alphabet_size=None, RNG=None, fixed=False, label=""):
        assert (not alphabet_size is None) != (not probabilities is None)
        self.name = "Alphabet"
        if not alphabet_size is None:
            self.alphabet_size = int(alphabet_size)
            if RNG is None:
                RNG = numpy.random.default_rng()
            self.probabilities = RNG.random(size=self.alphabet_size)
        else:
            self.probabilities = numpy.array(probabilities, numpy.float64)
            self.alphabet_size = self.probabilities.shape[0]
        self.probabilities /= numpy.sum(self.probabilities)
        where = numpy.where(self.probabilities > 0)[0]
        where2 = numpy.where(self.probabilities == 0)[0]
        self.log_probs = numpy.full(self.alphabet_size, -numpy.inf, numpy.float64)
        self.log_probs[where] = numpy.log(self.probabilities[where])
        self.tallies = numpy.zeros(self.alphabet_size)
        self.index = None
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = False
        self.dtype = numpy.int32
        return

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        if issubclass(obs.dtype.type, numpy.integer):
            if 'nonlog' in kwargs:
                probs = self.probabilities[obs[start:end]]
            else:
                probs = self.log_probs[obs[start:end]]
        else:
            probs = (numpy.sum(obs[start:end, :, :] *
                               self.probabilities.reshape(1, -1, 1), axis=1) /
                     numpy.sum(obs[start:end, :, :], axis=1))
            if 'nonlog' not in kwargs:
                where = numpy.where(probs > 0)
                where2 = numpy.where(probs == 0)
                probs[where] = numpy.log(probs[where])
                probs[where2] = -numpy.inf
        return probs

    def get_parameters(self, log=True):
        if log:
            return {"probabilities": self.log_probs}
        else:
            return {"probabilities": self.probabilities}

    @classmethod
    def update_tallies(self, *args):
        (func, start, end, state_idx, dist_idx, mix_idx,
         obsDtype, probsShape, params, smm_map) = args
        obsN = probsShape[0]
        dprobs = params['probabilities']
        views = []
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((obsN,), dtype=obsDtype, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        names = obsDtype.names
        if mix_idx is None:
            tallies = numpy.bincount(obs[names[dist_idx]][start:end],
                                     weights=probs[start:end, state_idx, 4],
                                     minlength=dprobs.shape[0])
        else:
            index = params['index']
            mixN = params['mixN']
            views.append(SharedMemory(smm_map['mix_probs']))
            mixprobs = numpy.ndarray((obsN, mixN,), numpy.float64,
                                     buffer=views[-1].buf)
            tmpProbs = mixprob[start:end, index] * probs[start:end, state_idx, 4]
            tallies = numpy.bincount(obs[names[dist_idx]][start:end],
                                     weights=tmpProbs, minlength=dprobs.shape[0])
        for V in views:
            V.close()
        return state_idx, dist_idx, mix_idx, tallies

    @classmethod
    def update_tree_tallies(self, *args):
        (func, start, end, state_idx, totalShape, probsShape, params,
         smm_map) = args
        seqN, num_states, num_nodes, _ = probsShape
        obsN, num_node_states, num_nodes = totalShape
        dprobs = params['probabilities']
        views = []
        views.append(SharedMemory(smm_map['tree_seqs']))
        tree_seqs = numpy.ndarray(seqN, dtype=numpy.int32,
                                  buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['total']))
        total = numpy.ndarray(totalShape, dtype=numpy.float64,
                                  buffer=views[-1].buf)
        tallies = numpy.sum(numpy.sum(
            probs[start:end, state_idx, :, 4].reshape(-1, 1, num_nodes) *
            total[tree_seqs[start:end], :, :], axis=2), axis=0)
        for V in views:
            V.close()
        return state_idx, tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        self.probabilities[:] = self.tallies / numpy.sum(self.tallies)
        where = numpy.where(self.probabilities > 0)[0]
        self.log_probs.fill(-numpy.inf)
        self.log_probs[where] = numpy.log(self.probabilities[where])
        return

    def print(self, level=0):
        tmp = " ".join([f"{x}:{self.probabilities[x] * 100:0.1f}%" for x in range(self.alphabet_size)])
        return f"{' '*level}Alphabet-{self.index} {self.label}\n      {tmp}"

    def generate_emission(self, RNG=None):
        if RNG is None:
            return numpy.searchsorted(numpy.cumsum(self.probabilities), numpy.random.random())
        else:
            return numpy.searchsorted(numpy.cumsum(self.probabilities), RNG.random())


class EmissionSummingDistribution(EmissionDiscreteDistribution):
    """Summing-based hmm emission distribution"""

    def __init__(self, probabilities=None, RNG=None, fixed=False, label=""):
        self.name = "Summing"
        self.probabilities = probabilities.astype(numpy.float64)
        self.log_probs = numpy.full(self.probabilities.shape, -numpy.inf,
                                    numpy.float64)
        where = numpy.where(self.probabilities > 0)[0]
        self.log_probs[where] = numpy.log(self.probabilities[where])
        self.label = label
        self.fixed = True

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        shape = [1 for x in range(len(obs.shape))]
        shape[1] = self.probabilities.shape[0]
        # probs = numpy.sum(obs[start:end] * self.probabilities.reshape(shape),
        #                   axis=1)
        # where = numpy.where(probs > 0)
        # where2 = numpy.where(probs == 0)
        # probs[where] = numpy.log(probs[where])
        # probs[where2] = -numpy.inf
        probs = scipy.special.logsumexp(obs[start:end] + self.log_probs.reshape(shape), axis=1)
        return probs

    def print(self, level=0):
        tmp = " ".join([f"{x}:{self.probabilities[x] * 100:0.1f}%"
                        for x in range(self.probabilities.shape[0])])
        return f"{' '*level}Summing-{self.index} {self.label}\n      {tmp}"

    def get_parameters(self, **kwargs):
        return {"probabilities": self.probabilities}


class EmissionPoissonDistribution(EmissionDiscreteDistribution):
    """Poisson-based hmm emission distribution"""

    def __init__(self, mu=None, RNG=None, maxobs=None, fixed=False, label=""):
        self.name = "Poisson"
        if mu is None:
            if RNG is None:
                RNG = numpy.random.default_rng()
            self.mu = RNG.random()
        else:
            self.mu = float(mu)
        if maxobs is not None:
            factorial = numpy.zeros(0, numpy.float64)
        else:
            factorial = numpy.zeros(maxobs + 1, numpy.float64)
            for i in range(1, maxobs + 1):
                factorial[i] = factorial[i - 1] + numpy.log(i)
        self.index = None
        self.tallies = numpy.zeros(2, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = True
        return

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        poissonN = kwargs['poissonN']
        smm_map = kwargs['smm_map']
        views = []
        views.append(SharedMemory(smm_map['logFactK']))
        logFactK = numpy.ndarray(poissonN, dtype=numpy.float64,
                                 buffer=views[-1].buf)
        self.prob = obs * numpy.log(self.mu) - self.mu - logFactK[obs]
        self.prob = numpy.exp(self.prob)
        for V in views:
            V.close()
        return self.prob

    @classmethod
    def update_tallies(self, *args):
        (func, start, end, state_idx, dist_idx, mix_idx,
         obsDtype, probsShape, params, smm_map) = args
        obsN = probsShape[0]
        views = []
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((obsN,), dtype=obsDtype, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        names = obsDtype.names
        if mix_idx is None:
            tallies = numpy.zeros(2, numpy.float64)
            tallies[0] = numpy.sum(obs[names[dist_idx]][start:end] *
                                   probs[start:end, state_idx, 3])
            tallies[1] = numpy.sum(probs[start:end, state_idx, 3])
        else:
            index, mixN = params[-2:]
            views.append(SharedMemory(smm_map['mix_probs']))
            mixprobs = numpy.ndarray((obsN, mixN,), numpy.float64,
                                     buffer=views[-1].buf)
            tmpProbs = mixprobs[start:end, index] * probs[start:end, state_idx, 3]
            tallies[0] = numpy.sum(obs[names[dist_idx]][start:end] * tmpProbs)
            tallies[1] = numpy.sum(tmpProbs)
        for V in views:
            V.close()
        return state_idx, dist_idx, mix_idx, tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        self.mu = self.tallies[0] / self.tallies[1]
        return

    def print(self, level=0):
        return f"{' '*level}Poisson-{self.index} {self.label}\n{' '*level}Lambda = {self.mu}"

    def generate_emission(self, RNG=numpy.random):
        return RNG.poisson(lam=self.mu)

    def get_parameters(self):
        return numpy.array([self.mu], numpy.float64)


class EmissionDiscreteMixtureDistribution(EmissionDiscreteDistribution):
    """Combination of multiple discrete emission distributions"""

    def __init__(self, distributions, proportions=None, RNG=None, fixed=False, label=""):
        for D in distributions:
            assert issubclass(type(D), EmissionDiscreteDistribution)
        self.name = "DiscreteMixture"
        self.distributions = distributions
        if not proportions is None:
            assert len(distributions) == len(proportions)
            self.proportions = numpy.array(proportions, numpy.float64)
        else:
            if RNG is None:
                RNG = numpy.random.default_rng()
            self.proportions = RNG.random(size=len(distributions))
        self.proportions /= numpy.sum(self.proportions)
        self.num_distributions = len(self.distributions)
        self.distribution_indices = []
        self.index = None
        self.tallies = numpy.zeros(self.num_distributions, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = True
        return

    def score_observations(self, obs, *kwargs):
        start = kwargs['start']
        end = kwargs['end']
        mixN, = kwargs['mixN']
        smm_map = kwargs['smm_map']
        views = []
        views.append(SharedMemory(smm_map['mix_probs']))
        mix_probs = numpy.ndarray((obs.shape[0], mixN,), numpy.float64,
                                  buffer=views[-1].buf)
        probs = None
        for i, index in enumerate(self.distribution_indices):
            if probs is None:
                mix_probs[start:end, index] *= self.proportions[i]
                probs = numpy.copy(mix_probs[start:end, index])
            else:
                probs *= mix_probs[start:end, index]
        sums = numpy.sum(mix_probs[start:end, self.distribution_indices],
                         axis=1)
        where = numpy.where(sums > 0)[0]
        for i in self.distribution_indices:
            mix_probs[where + start, i] /= sums[where]
        for V in views:
            V.close()
        return probs

    def clear_tallies(self):
        super(EmissionDiscreteMixtureDistribution, self).clear_tallies()
        for D in self.distributions:
            D.clear_tallies()
        return

    @classmethod
    def update_tallies(self, *args):
        (func, start, end, state_idx, dist_idx, mix_idx,
         obsDtype, probsShape, params, smm_map) = args
        proporions, _, distribution_indices, mixN = params
        obsN = probsShape[0]
        views = []
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['mix_probs']))
        mixprobs = numpy.ndarray((obsN, mixN,), numpy.float64,
                                 buffer=views[-1].buf)
        names = obsDtype.names
        tallies = numpy.zeros(distribution_indices.shape[0], numpy.float64)
        tmpProbs = (mixprobs[start:end, distribution_indices] *
                    probs[start:end, state_idx, 3:4])
        tallies = numpy.sum(tmpProbs, axis=0)
        for V in views:
            V.close()
        return state_idx, dist_idx, mix_idx, tallies


    def apply_tallies(self):
        for D in self.distributions:
            D.apply_tallies()
        if self.updated:
            return
        self.updated = True
        self.proportions[:] = self.tallies
        self.proportions /= numpy.sum(self.proportions)
        return

    def print(self, level=0):
        output = []
        output.append(f"{' '*level}Mixture Model-{self.index} {self.label}")
        for i in range(len(self.distributions)):
            output.append(f"{' '*level}{self.proportions[i] * 100:0.1f}%")
            output.append(self.distributions[i].print(level + 1))
        output = "\n".join(output)
        return output

    def generate_emission(self, RNG=numpy.random):
        return self.distributions[numpy.searchsorted(numpy.cumsum(
            self.proportions), RNG.random())].generate_emission(RNG)

    def get_parameters(self):
        return self.proportions, [D.get_parameters() for D in self.distributions]


class EmissionContinuousDistribution(EmissionDistribution):
    """Continuous-based hmm emission distribution"""


class EmissionGaussianDistribution(EmissionContinuousDistribution):
    """Gaussian-based hmm emission distribution"""
    pdf_constant = (numpy.pi * 2) ** -0.5
    log_pdf_constant = numpy.log(numpy.pi * 2) * -0.5

    def __init__(self, mu=None, sigma=None, RNG=None, fixed=False, label=""):
        self.name = "Gaussian"
        if RNG is None:
            RNG = numpy.random.default_rng()
        if mu is None:
            self.mu = RNG.random()
        else:
            self.mu = float(mu)
        if sigma is None:
            self.sigma = RNG.random()
        else:
            self.sigma = float(sigma)
        self.logsigma = numpy.log(self.sigma)
        self.index = None
        self.tallies = numpy.zeros(3, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = False
        self.dtype = numpy.float64
        return

    def score_observations(self, obs, **kwargs):
        # start = kwargs['start']
        # end = kwargs['end']
        # prob = (self.log_pdf_constant - self.logsigma - 0.5 *
        #         ((obs[start:end] - self.mu) / self.sigma) ** 2)
        prob = scipy.stats.norm.logpdf(obs, self.mu, self.sigma)
        return prob

    @classmethod
    def update_tallies(self, **kwargs):
        smm_map = kwargs['smm_map']
        params = kwargs['params']
        obsDtype = kwargs['obsDtype']
        node_name = kwargs['node_name']
        state_idx = kwargs['state_idx']
        dist_idx = kwargs['dist_idx']
        mix_idx = kwargs['mix_idx']
        start = kwargs['start']
        end = kwargs['end']
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(5, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_mixdists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((num_seqs, num_nodes,), dtype=obsDtype,
                            buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['total']))
        total = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        mu = params['mu']
        names = obsDtype.names
        tallies = numpy.zeros(3, numpy.float64)
        if mix_idx is None:
            tallies[0] += numpy.sum(
                obs[names[dist_idx]][start:end, node_idx] *
                total[start:end, state_idx, node_idx])
            tallies[1] += numpy.sum(
                (obs[names[dist_idx]][start:end, node_idx] - mu) ** 2 *
                total[start:end, state_idx, node_idx])
            tallies[2] += numpy.sum(
                total[start:end, state_idx, node_idx])
        else:
            mix_indices = kwargs['mix_indices']
            views.append(SharedMemory(smm_map['mix_probs']))
            mix_probs = numpy.ndarray((num_seqs, num_mixdists), numpy.float64,
                                  buffer=views[-1].buf)
            proportions = params['proportions']
            mix_proportions = numpy.exp(mix_probs[start:end, mix_indices] -
                                        numpy.amax(mix_probs[start:end, mix_indices],
                                                   axis=1, keepdims=True))
            mix_proportions *= proportions.reshape(1, -1)
            mix_proportions /= numpy.sum(mix_proportions, axis=1, keepdim=True)
            probs = total[start:end, state_idx, node_idx] * mix_proportions[:, mix_idx]
            tallies[0] += numpy.sum(
                obs[names[dist_idx]][start:end, node_idx] * probs)
            tallies[1] += numpy.sum(
                (obs[names[dist_idx]][start:end, node_idx] - mu) ** 2 * probs)
            tallies[2] += numpy.sum(probs)
        for V in views:
            V.close()
        return node_name, state_idx, dist_idx, mix_idx, tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        self.mu = self.tallies[0] / self.tallies[2]
        self.sigma = (self.tallies[1] / self.tallies[2]) ** 0.5
        self.logsigma = numpy.log(self.sigma)
        return

    def print(self, level=0):
        return f"{' '*level}Normal-{self.index} {self.label}\n" \
               f"{' '*level}Mu = {self.mu}\n" \
               f"{' '*level}Sigma = {self.sigma}"

    def generate_emission(self, RNG):
        return RNG.normal(loc=self.mu, scale=self.sigma)

    def get_parameters(self, log=None):
        return {'mu': self.mu, 'sigma':self.sigma}


class EmissionLogNormalDistribution(EmissionContinuousDistribution):
    """Log-Gaussian-based hmm emission distribution"""
    pdf_constant = (numpy.pi * 2) ** -0.5

    def __init__(self, mu=None, sigma=None, RNG=None, fixed=False, label=""):
        self.name = "LogNormal"
        if RNG is None:
            RNG = numpy.random.default_rng()
        if mu is None:
            self.mu = RNG.random()
        else:
            self.mu = float(mu)
        if sigma is None:
            self.sigma = RNG.random()
        else:
            self.sigma = float(sigma)
        self.index = None
        self.tallies = numpy.zeros(3, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = True
        return

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        prob = numpy.zeros(end - start, numpy.float64)
        valid = numpy.where(obs[start:end] > 0)[0]
        prob[valid] = (self.pdf_constant / (obs[valid + start] * self.sigma) *
                       numpy.exp(-0.5 * ((numpy.log(obs[valid + start]) - self.mu)
                                         / self.sigma) ** 2))
        return prob

    @classmethod
    def update_tallies(self, *args):
        (func, start, end, state_idx, dist_idx, mix_idx,
         obsDtype, probsShape, params, smm_map) = args
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(4, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((num_seqs, num_nodes,), dtype=obsDtype,
                            buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray((num_seqs, num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        names = obsDtype.names
        tallies = numpy.zeros(3, numpy.float64)
        if mix_idx is None:
            tallies[0] = numpy.sum(obs[names[dist_idx]][start:end] *
                                   probs[start:end, state_idx, 3])
            tallies[1] = numpy.sum((obs[names[dist_idx]][start:end] - params[0]) ** 2 *
                                   probs[start:end, state_idx, 3])
            tallies[2] = numpy.sum(probs[start:end, state_idx, 3])
        else:
            index, mixN = params[-2:]
            views.append(SharedMemory(smm_map['mixture_probs']))
            mixprobs = numpy.ndarray((obsN, mixN,), numpy.float64,
                                     buffer=views[-1].buf)
            tmpProbs = mixprobs[start:end, index] * probs[start:end, state_idx, 3]
            tallies[0] = numpy.sum(obs[names[dist_idx]][start:end] * tmpProbs)
            tallies[1] = numpy.sum((obs[names[dist_idx]][start:end] - params[0]) ** 2 *
                                   tmpProbs)
            tallies[2] = numpy.sum(tmpProbs)
        for V in views:
            V.close()
        return state_idx, dist_idx, mix_idx, tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        self.mu = self.tallies[0] / self.tallies[2]
        self.sigma = (self.tallies[1] / self.tallies[2]) ** 0.5
        return

    def print(self, level=0):
        return f"{' '*level}LogNormal-{self.index} {self.label}\n" \
               f"{' '*level}Mu = {self.mu}\n" \
               f"{' '*level}Sigma = {self.sigma}"

    def generate_emission(self, RNG=None):
        if RNG is None:
            RNG = numpy.random
        return numpy.log(RNG.normal(loc=self.mu, scale=self.sigma))

    def get_parameters(self):
        return numpy.array([self.mu, self.sigma], numpy.float64)


class EmissionGammaDistribution(EmissionContinuousDistribution):
    """Gamma-based hmm emission distribution"""

    def __init__(self, mu=None, sigma=None, RNG=None, fixed=False, label=""):
        self.name = "Gamma"
        if RNG is None:
            RNG = numpy.random.default_rng()
        if mu is None:
            mu = RNG.random()
        else:
            mu = float(mu)
        if sigma is None:
            sigma = RNG.random()
        else:
            sigma = float(sigma)
        self.k = mu ** 2 / sigma
        self.theta = sigma / mu
        self.log_denom = scipy.special.loggamma(self.k) + numpy.log(self.theta) * self.k
        self.index = None
        self.tallies = numpy.zeros(3, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = True
        return

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        prob = numpy.exp(numpy.log(obs ** (self.k - 1) *
                         numpy.exp(-obs[start:end] / self.theta)) - self.log_denom)
        return prob

    @classmethod
    def update_tallies(self, *args):
        (func, start, end, state_idx, dist_idx, mix_idx,
         obsDtype, probsShape, params, smm_map) = args
        obsN = probsShape[0]
        views = []
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((obsN,), dtype=obsDtype, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        names = obsDtype.names
        tallies = numpy.zeros(3, numpy.float64)
        if mix_idx is None:
            tallies[0] = numpy.sum(obs[names[dist_idx]][start:end] *
                                   probs[start:end, state_idx, 3])
            tallies[1] = numpy.sum((obs[names[dist_idx]][start:end] - params[0]) ** 2 *
                                   probs[start:end, state_idx, 3])
            tallies[2] = numpy.sum(probs[start:end, state_idx, 3])
        else:
            index, mixN = args[-2:]
            views.append(SharedMemory(smm_map['mixture_probs']))
            mixprobs = numpy.ndarray((obsN, mixN,), numpy.float64,
                                     buffer=views[-1].buf)
            tmpProbs = mixprobs[start:end, index] * probs[start:end, state_idx, 3]
            tallies[0] = numpy.sum(obs[names[dist_idx]][start:end] * tmpProbs)
            tallies[1] = numpy.sum((obs[names[dist_idx]][start:end] - params[0]) ** 2 *
                                   tmpProbs)
            tallies[2] = numpy.sum(tmpProbs)
        for V in views:
            V.close()
        return state_idx, dist_idx, mix_idx, tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        mu = self.tallies[0] / self.tallies[2]
        sigma = (self.tallies[1] / self.tallies[2]) ** 0.5
        self.k = mu ** 2 / sigma
        self.theta = sigma / mu
        self.log_denom = scipy.special.loggamma(self.k) + numpy.log(self.theta) * self.k
        return

    def print(self, level=0):
        return f"{' '*level}Gamma-{self.index} {self.label}\n" \
               f"{' '*level}Mu = {self.k * self.theta}\n" \
               f"{' '*level}Sigma = {self.k * self.theta ** 2}"

    def generate_emission(self, RNG=numpy.random):
        return RNG.gamma(shape=self.k, scale=self.theta)

    def get_parameters(self):
        return numpy.array([self.k, self.theta], numpy.float64)


class EmissionZeroDistribution(EmissionContinuousDistribution):
    """Zero for zero-inflated hmm emission distributions"""

    def __init__(self, fixed=False, label="", **kwargs):
        self.name = "Zero"
        self.index = None
        self.tallies = numpy.zeros(0, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = True
        return

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        prob = numpy.full(obs.shape[0], -numpy.inf, numpy.float64)
        prob[numpy.where(obs == 0)] = 0
        return prob

    @classmethod
    def update_tallies(self, **kwargs):
        node_name = kwargs['node_name']
        state_idx = kwargs['state_idx']
        dist_idx = kwargs['dist_idx']
        mix_idx = kwargs['mix_idx']
        tallies = numpy.zeros(0, numpy.float64)
        return node_name, state_idx, dist_idx, mix_idx, tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        return

    def print(self, level=0):
        return f"{' '*level}Zero-{self.index} {self.label}"

    def generate_emission(self, RNG):
        return numpy.zeros(1, numpy.float64)

    def get_parameters(self, **kwargs):
        return {}


class EmissionContinuousMixtureDistribution(EmissionContinuousDistribution):
    """Combination of multiple continuous emission distributions"""

    def __init__(self, distributions, proportions=None, RNG=None, fixed=False, label=""):
        self.name = "ContinuousMixture"
        if RNG is None:
            RNG = numpy.random.default_rng()
        for D in distributions:
            assert issubclass(type(D), EmissionContinuousDistribution)
        self.distributions = distributions
        if not proportions is None:
            assert len(distributions) == len(proportions)
            self.proportions = numpy.array(proportions, numpy.float64)
        else:
            self.proportions = RNG.random(size=len(distributions))
        self.proportions /= numpy.sum(self.proportions)
        self.log_proportions = numpy.log(self.proportions)
        self.num_distributions = len(self.distributions)
        self.distribution_indices = []
        self.index = None
        self.tallies = numpy.zeros(self.num_distributions, numpy.float64)
        self.fixed = bool(fixed)
        self.label = str(label)
        self.updated = True
        return

    def score_observations(self, obs, **kwargs):
        start = kwargs['start']
        end = kwargs['end']
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(5, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_mixdists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['mix_probs']))
        mix_probs = numpy.ndarray((num_seqs, num_mixdists), numpy.float64,
                                  buffer=views[-1].buf)
        probs = scipy.special.logsumexp(mix_probs[start:end, self.distribution_indices] +
                                        self.log_proportions.reshape(-1, 1), axis=1)
        for V in views:
            V.close()
        return probs

    def clear_tallies(self):
        super(EmissionContinuousMixtureDistribution, self).clear_tallies()
        for D in self.distributions:
            D.clear_tallies()
        return

    @classmethod
    def update_tallies(self, **kwargs):
        smm_map = kwargs['smm_map']
        params = kwargs['params']
        node_name = kwargs['node_name']
        state_idx = kwargs['state_idx']
        dist_idx = kwargs['dist_idx']
        mix_idx = kwargs['mix_idx']
        start = kwargs['start']
        end = kwargs['end']
        proportions = params['proportions']
        dist_indices = params['dist_indices']
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = numpy.ndarray(5, numpy.int64, buffer=views[-1].buf)
        num_nodes, num_dists, num_mixdists, num_states, num_seqs = sizes
        views.append(SharedMemory(smm_map['mix_probs']))
        mix_probs = numpy.ndarray((num_seqs, num_mixdists), numpy.float64,
                              buffer=views[-1].buf)
        mix_proportions = numpy.exp(mix_probs[start:end, dist_indices] -
                                    numpy.amax(mix_probs[start:end, dist_indices],
                                               axis=1, keepdims=True))
        mix_proportions *= proportions.reshape(1, -1)
        mix_proportions /= numpy.sum(mix_proportions, axis=1, keepdim=True)
        tallies = numpy.sum(mix_proportions, axis=0)
        for V in views:
            V.close()
        return node_name, state_idx, dist_idx, mix_idx, tallies

    def apply_tallies(self):
        for D in self.distributions:
            D.apply_tallies()
        if self.updated:
            return
        self.updated = True
        self.proportions[:] = self.tallies
        self.proportions /= numpy.sum(self.proportions)
        return

    def print(self, level=0):
        output = []
        output.append(f"{' '*level}Mixture Model-{self.index} {self.label}")
        for i in range(len(self.distributions)):
            output.append(f"{' '*level}{self.proportions[i] * 100:0.1f}%")
            output.append(self.distributions[i].print(level + 1))
        output = "\n".join(output)
        return output

    def generate_emission(self, RNG):
        return self.distributions[numpy.searchsorted(numpy.cumsum(
            self.proportions), RNG.random())].generate_emission(RNG)

    def get_parameters(self, **kwargs):
        return {'proportions': self.proportions}#, [D.get_parameters() for D in self.distributions]













