from torch.nn.modules import Module
from torch.nn import Parameter

from poissonmle import *


class MyPoisson:
    def __init__(self, a_lambda):
        self.a_lambda = a_lambda

    def compute_prob(self, k):
        nominator = (self.a_lambda ** k) * torch.exp(- self.a_lambda)
        denominator = torch.lgamma(k + 1).exp()
        prob = nominator / denominator
        return prob

    def compute_log_prob(self, k):
        prob = self.compute_prob(k)
        return torch.log(prob)


class LogLikelihoodPoisson(Module):
    def __init__(self, current_lambda):
        super(LogLikelihoodPoisson, self).__init__()
        self.current_lambda = Parameter(data=current_lambda, requires_grad=True)

    def forward(self, vector_of_observations):
        # compute_sum_log_likelihood
        poisson_rv_instance = MyPoisson(a_lambda=self.current_lambda)
        res = poisson_rv_instance.compute_log_prob(k=vector_of_observations)
        return res.sum()


class PoissonOptimizer:
    def __init__(self, x):
        self.x = x

    def compute_sum_log_likelihood(self, model):
        '''
        sum over i,
        log{ P(N_1 = n_1 |lambda) * P(N_2 = n_2 |lambda) * ... * P(N_i = n_i |lambda) * ... * P(N_n = n_n |lambda)}
        '''
        return - model(self.x)

    def create_model_object(self):
        lambda_initiator = torch.tensor(1.0)
        claim_arrival_likelihood = LogLikelihoodPoisson(current_lambda=lambda_initiator)
        return claim_arrival_likelihood

    def optimize(self):
        model = self.create_model_object()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        for t in range(100):
            print(t)
            loss = self.compute_sum_log_likelihood(model)
            loss.backward()
            optimizer.step( lambda: self.compute_sum_log_likelihood(model))
            optimizer.zero_grad()
            print("-" * 50)
            print("likelihood = {}".format(loss.data))
            print("learned lambda = {}".format(list(model.parameters())[0].data))

        optimization_res = optimizer.param_groups[0]['params'][0]
        return optimization_res

    def conduct_optimization(self):
        optimization_res = self.optimize()
        print('  ')
        print('  ')
        # print('optimization_res :', optimization_res)
        return optimization_res


if __name__ == '__main__':
    # compute probability of poisson r.v
    # poisson_rv = MyPoisson(a_lambda=torch.tensor(data=[4.0]))
    # res = poisson_rv.compute_prob(torch.tensor(data=[4.0]))
    # print(res)

    # compute the log likelihood of poisson
    # observations = torch.tensor(data=[4.0, 5.0, 2.0, 5.0, 3.0])
    # log_likelihood_poisson = LogLikelihoodPoisson(current_lambda=torch.tensor(data=[4.0]))
    # sum_log_likelihood = log_likelihood_poisson.forward(vector_of_observations=observations)
    # print(sum_log_likelihood)

    # compare two estimation process, mle and automatic gradient
    given_sample_size = 250
    given_poisson_parameter = 4
    samples = generate_poisson_random_variables(poisson_parameter=given_poisson_parameter,
                                                sample_size=given_sample_size)

    poisson_optimizer = PoissonOptimizer(x=samples)
    parameter_acquire_by_optimization_process = poisson_optimizer.conduct_optimization()
    the_mle_estimator = estimate_poisson_parameter_with_mle(samples)

    print('given_poisson_parameter :', given_poisson_parameter)
    print('parameter acquire by optimization process :', parameter_acquire_by_optimization_process.data)
    print('mle_estimator :', the_mle_estimator)