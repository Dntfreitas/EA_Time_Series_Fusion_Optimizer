def decode_chromosome_or_particle(chromosome_or_particle, verbose=True):
    """
Receives an encoded chromosome/particle and decodes the chromosome/particle, returning its specification
    @param chromosome_or_particle:the chromosome/particle to be decoded (should be a NumPy array)
    @param verbose:whether to print the chromosome in a human-readable manner
    @return:the parameters of the decoded chromosome
    """
    # Codification of the chromosome
    no_channels = {'000': 'Fp2–F4', '001': 'C4–A1', '010': 'F4–C4', '011': 'Fp2–F4 and C4–A1',
                   '100': 'Fp2–F4 and F4–C4', '101': 'F4–C4 and C4–A1', '110': 'Fp2–F4, F4–C4, and C4–A1',
                   '111': 'Fp2–F4, F4–C4, and C4–A1'}
    no_time_steps = {'00': 10, '01': 15, '10': 20, '11': 25}
    lstm_layers = {'0': 1, '1': 2}
    lstm_type = {'0': 'LSTM', '1': 'BLSTM'}
    lstm_layers_shape = {'00': 100, '01': 200, '10': 300, '11': 400}
    dropout_percentage = {'00': '0 %', '01': '5 %', '10': '10 %', '11': '15 %'}
    dense_layers_shape = {'00': 0, '01': 200, '10': 300, '11': 400}
    dense_layers_activation_function = {'00': 'tanh', '01': 'sigmoid', '10': 'relu', '11': 'selu'}
    # Convert the chromosome to string
    chromosome_str = ''
    for locus in chromosome_or_particle:
        chromosome_str += str(locus)
    # Decode the chromosome
    no_channels_decoded = no_channels[chromosome_str[0:3]]
    no_time_steps_decoded = no_time_steps[chromosome_str[3:5]]
    lstm_layers_decoded = lstm_layers[chromosome_str[5]]
    lstm_type_decoded = lstm_type[chromosome_str[6]]
    lstm_layers_shape_decoded = lstm_layers_shape[chromosome_str[7:9]]
    dropout_percentage_decoded = dropout_percentage[chromosome_str[9:11]]
    dense_layers_shape_decoded = dense_layers_shape[chromosome_str[11:13]]
    dense_layers_activation_function_decoded = dense_layers_activation_function[chromosome_str[13:15]]
    # Print the decoded the chromosome
    if verbose:
        print('Chromosome:', chromosome_or_particle)
        print('Number of channels to be fused:', no_channels_decoded)
        print('Number of time steps to be considered by the LSTM:', no_time_steps_decoded)
        print('Number of LSTM layers for each channel:', lstm_layers_decoded)
        print('Type of LSTM:', lstm_type_decoded)
        print('Shape of the LSTM layers:', lstm_layers_shape_decoded)
        print('Percentage of dropout for the recurrent and dense layers:', dropout_percentage_decoded)
        print('Shape of the dense layers:', dense_layers_shape_decoded)
        print('Activation function for the dense layers:', dense_layers_activation_function_decoded)
    # Return the results
    return no_channels_decoded, no_time_steps_decoded, lstm_layers_decoded, lstm_type_decoded, lstm_layers_shape_decoded, dropout_percentage_decoded, dense_layers_shape_decoded, dense_layers_activation_function_decoded
