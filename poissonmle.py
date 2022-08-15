import torch
from torch.distributions.poisson import Poisson


def generate_poisson_random_variables(poisson_parameter, sample_size):
    poisson_instance = Poisson(poisson_parameter, validate_args=None)
    samples_vec = poisson_instance.sample(sample_shape=torch.Size([1, sample_size]))
    return samples_vec


def estimate_poisson_parameter_with_mle(observations):
    mle_estimator = torch.mean(observations)
    return mle_estimator


if __name__ == '__main__':
    given_sample_size = 250
    given_poisson_parameter = 4
    samples = generate_poisson_random_variables(poisson_parameter=given_poisson_parameter,
                                                sample_size=given_sample_size)
    the_mle_estimator = estimate_poisson_parameter_with_mle(samples)
    print('given_poisson_parameter :', given_poisson_parameter)
    print('mle_estimator :', the_mle_estimator)
