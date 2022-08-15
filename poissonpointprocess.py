from torch.distributions.poisson import Poisson
from torch.nn.modules import Module
from torch.nn import Parameter
import torch


def generate_poisson_parameters_according_to_a_model(x: list, c: int, d: int):
    parameters_list = []
    for i in x:
        parameters_list.append(c*i + d)
    return torch.tensor(parameters_list)


def generate_samples(parameters, sample_size):
    poisson_instance = Poisson(parameters, validate_args=None)
    samples_vec = poisson_instance.sample(sample_shape=torch.Size([1, sample_size]))
    return samples_vec


class MyPoisson:
    def __init__(self, c, d):
        self.c = c
        self.d = d

    def compute_prob(self, k):
        time_vec = torch.tensor([0.25, 0.5, 0.75, 1])
        poisson_parameter = self.c * time_vec + self.d
        nominator = (poisson_parameter ** k) * torch.exp(- poisson_parameter)
        denominator = torch.lgamma(k + 1).exp()
        prob = nominator / denominator
        return prob

    def compute_log_prob(self, k):
        prob = self.compute_prob(k)
        return torch.log(prob)


class LogLikelihoodPoisson(Module):
    def __init__(self, c, d):
        super(LogLikelihoodPoisson, self).__init__()
        self.current_c = Parameter(data=c, requires_grad=True)
        self.current_d = Parameter(data=d, requires_grad=True)

    def forward(self, vector_of_observations):
        # compute_sum_log_likelihood
        poisson_rv_instance = MyPoisson(c=self.current_c, d=self.current_d)
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
        c_initiator = torch.tensor(1.0)
        d_initiator = torch.tensor(1.0)
        claim_arrival_likelihood = LogLikelihoodPoisson(c=c_initiator, d=d_initiator)
        return claim_arrival_likelihood

    def optimize(self):
        model = self.create_model_object()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        for t in range(100):
            print('iteration :', t)
            loss = self.compute_sum_log_likelihood(model)
            loss.backward()
            optimizer.step( lambda: self.compute_sum_log_likelihood(model))
            optimizer.zero_grad()
            print("-" * 50)
            print("likelihood = {}".format(loss.data))
            print("learned c = {}".format(list(model.parameters())[0].data))
            print("learned d = {}".format(list(model.parameters())[1].data))

        # optimization_res = optimizer.param_groups[0]['params'][0]
        optimization_res = optimizer.param_groups[0]['params']
        return optimization_res

    def conduct_optimization(self):
        optimization_res = self.optimize()
        print('  ')
        print('  ')
        # print('optimization_res :', optimization_res)
        return optimization_res


if __name__ == '__main__':
    given_c = 4
    given_d = 1
    generated_parameters = generate_poisson_parameters_according_to_a_model(x=[0.25, 0.5, 0.75, 1], c=given_c,
                                                                            d=given_d)
    # print(generated_parameters)

    given_sample_size = 1000
    generated_samples = generate_samples(parameters=generated_parameters, sample_size=given_sample_size)
    # print(generated_samples)

    poisson_optimizer = PoissonOptimizer(x=generated_samples)
    parameter_acquire_by_optimization_process = poisson_optimizer.conduct_optimization()
    print('given_poisson_parameter :')
    print('c :', given_c)
    print('d :', given_d)

    print(' ')
    print(' ')

    print('parameter_acquire_by_optimization_process')
    print('c :', parameter_acquire_by_optimization_process[0].data)
    print('c :', parameter_acquire_by_optimization_process[1].data)
