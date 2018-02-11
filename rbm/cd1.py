import tensorflow as tf

# This is the gradient function
def cd1(inputs, rbm_weights):
    visible_data_sampled = tf.cast(tf.greater_equal(inputs, tf.random_uniform(inputs.shape)), "float32")

    # visible_data = sample_bernoulli(visible_data);

    # visible_data_shape = [visible_data.shape[0].value, 1]
    # hidden_states_shape = [rbm_weights.shape[0].value, 1]

    # Visible state to hidden probabilities
    ones = tf.constant(1, "float32", [rbm_weights.shape[0], visible_data_sampled.shape[1]])
    multiply_term = tf.exp(tf.matmul(tf.negative(rbm_weights), visible_data_sampled))

    hidden_probabilities = tf.div(ones, tf.add(ones, multiply_term))

    # hidden_probability = 1. / (1 + exp(-rbm_w * visible_state));

    # hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);

    hidden_states = tf.cast(tf.greater_equal(hidden_probabilities,  tf.random_uniform(hidden_probabilities.shape)), "float32")

    # hidden_states = sample_bernoulli(hidden_probabilities);

    # Compute the hidden state to visible probabilities
    ones_hidden = tf.constant(1, "float32", inputs.shape)
    multiply_term_hidden = tf.exp(tf.matmul(tf.transpose(tf.negative(rbm_weights)), hidden_states))
    visible_probabilities = tf.div(ones_hidden, tf.add(ones_hidden, multiply_term_hidden))

    # visible_probability = 1. / (1 + exp(-rbm_w' * hidden_state));

    # visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_states);

    visible_states = tf.cast(tf.greater_equal(visible_probabilities, tf.random_uniform(inputs.shape)), "float32")

    # visible_states = sample_bernoulli(visible_probabilities);|

    # Use probabilities directly in improvement

    hidden_probabilities2 = tf.div(ones, tf.add(ones, tf.exp(tf.matmul(tf.negative(rbm_weights), visible_states))))

    # hidden_probability = 1 ./ (1 + exp(-rbm_w * visible_state));

    # hidden_probabilities2 = visible_state_to_hidden_probabilities(rbm_w, visible_states);

    # hidden_states2 = tf.cast(tf.greater_equal(hidden_probabilities2,  tf.random_uniform(hidden_probabilities.shape)), "float32")

    # hidden_states2 = sample_bernoulli(hidden_probabilities2);

    numberOfConfigurations = tf.constant(inputs.shape[1].value, dtype = "float32")

    term1 = tf.div(tf.matmul(hidden_states, tf.transpose(inputs)), numberOfConfigurations)
    term2 = tf.div(tf.matmul(hidden_probabilities2, tf.transpose(visible_states)), hidden_probabilities2.shape[1].value)

    return tf.subtract(term1, term2)


    # ret = configuration_goodness_gradient(visible_data, hidden_states) - configuration_goodness_gradient(visible_states, hidden_probabilities2);





# def config_gradient():

    # d_G_by_rbm_w = hidden_state * visible_state / size(hidden_state, 2);