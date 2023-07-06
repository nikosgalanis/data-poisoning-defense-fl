import tensorflow as tf
import tensorflow_federated as tff
import random
from tqdm import tqdm


def train_and_eval(model, n_train_epochs, ):

    # lists to hold the metrics that we want to compute
    accs = []
    losses = []
    class_recall = []

    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    preprocessed_example_dataset = preprocess(example_dataset, True)
    
        
    # build the process to have the model's architecture
    evaluation_process = tff.learning.algorithms.build_fed_eval(mnist_model)

    # initialize the state of the evaluation
    sample_test_clients = emnist_test.client_ids[0:n_test_clients]


    federated_test_data = make_federated_data(emnist_test, sample_test_clients, 0, 0, train=False)

    # fix the random clients so that they are the same for every model
    clients = []

    for i in range(n_train_epochs):
        clients.append(random.sample(emnist_train.client_ids, n_clients))

    for i in [0, 1, 2, 3, 4, 5]:
        evaluation_state = evaluation_process.initialize()

        eval_acc = []
        eval_loss = []
        eval_recall = []

        mal_users_percentage = i / 10
        
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        training_process = tff.learning.algorithms.build_unweighted_fed_avg(
            model,
            client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate = client_learning_rate),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate = server_learning_rate))

        train_state = training_process.initialize()
        print("Model with " + str(i * 10) + "% mal clients")
        for epoch in tqdm(range(n_train_epochs), position = 0, leave = True):
            # clients = emnist_train.client_ids[0:n_clients]

            federated_train_data = make_federated_data(emnist_train, clients[epoch], target_value, poisoned_value, train=True, mal_users_percentage=mal_users_percentage)      
            
            # run a next on the training process to train the model
            result = training_process.next(train_state, federated_train_data)
            # update the model's state and get access to the metrics
            train_state = result.state
            
            train_metrics = result.metrics
            # print the training metrics
            
            # get weights from the trainged model
            model_weights = training_process.get_model_weights(train_state)
            # update the evaluation state with them
            evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
            # run a next() to evaluate the model
            evaluation_output = evaluation_process.next(evaluation_state, federated_test_data)

            # get access to the evaluation metrics
            eval_metrics = evaluation_output.metrics['client_work']['eval']['total_rounds_metrics']

            eval_acc.append(eval_metrics['sparse_categorical_accuracy'])
            eval_loss.append(eval_metrics['loss'])
            eval_recall.append(eval_metrics['specific_class_recall'])

        accs.append(eval_acc)
        losses.append(eval_loss)
        class_recall.append(eval_recall)