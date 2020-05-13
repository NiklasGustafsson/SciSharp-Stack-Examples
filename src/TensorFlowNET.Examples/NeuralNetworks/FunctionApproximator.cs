using System;
using System.Collections.Generic;
using System.Net.NetworkInformation;
using System.Diagnostics;
using System.Linq;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.NeuralNetworks
{
    /// <summary>
    /// Based on the F# function approximation sample, this example uses a shallow neural network to estimate a scalar 
    /// real-valued function. It is trained on a set of N input values chosen from a particular range of R, and f(x) 
    /// values computed from the actual function.
    /// </summary>
    public class FunctionApproximator : SciSharpExample, IExample
    {
        const int training_size = 64*1024;     // Number of points for training function.
        const int test_size = 256;             // Number of data points in the validation set.

        // These are all values that are passed between stages of the example construction and execution.
        NDArray X_train = null;
        NDArray Y_train = null;

        NDArray X_test = null;
        NDArray Y_test = null;

        Tensor input = null;
        Tensor y = null;
        Tensor y_hat = null;

        Tensor loss_op = null;
        Tensor global_step = null;
        Operation train_op = null;

        public ExampleConfig InitConfig() =>
            Config = new ExampleConfig
            {
                Name = "Function Approximator Network",
                Enabled = true,
                IsImportingGraph = false
            };

        const float x_min = 1.0f;      // Min of the range of input data
        const float x_max = 15.0f;     // Max of the range of input data

        const float learning_rate = 0.00001f;
        const int training_epochs = 150000;

        // Size of hidden layers
        const int n_hidden_layer_1 = 25; // Hidden layer 1
        const int n_hidden_layer_2 = 45; // Hidden layer 2

        /// <summary>
        /// Generate an array of values, uniformly spaced from the min to the max.
        /// </summary>
        /// <param name="points">The number of data points to generate.</param>
        /// <returns></returns>
        private static IEnumerable<float> linspace(float min, float max, int points)
        {
            return Enumerable.Range(0, points).Select(i => min + (max - min) / (float)points * (float)i);
        }

        private Func<float, float> func = (x) => (float)((20.0f * x + 3.0f * System.Math.Pow(x, 2.0f) + 0.1f * System.Math.Pow(x, 3.0f)) * System.Math.Sin(x) * System.Math.Exp(-0.1f * x));

        /// <summary>
        /// Create the training and validation data sets
        /// </summary>
        public override void PrepareData()
        {
            const float noise_mean = 0.0f; // Mean of the Gaussian noise adder
            const float noise_sd = 10.0f;  // Std.Dev of the Gaussian noise adder

            {
                // Create a training data set.
                var X_raw = linspace(x_min, x_max, training_size);
                var Y_raw = X_raw.Select(func);
                X_train = np.array(X_raw).reshape(training_size, 1);
                Y_train = np.array(Y_raw) + np.random.normal(noise_mean, noise_sd, training_size);
            }

            {
                var X_raw = np.random.uniform(x_min, x_max, test_size);
                var Y_raw = X_raw.ToArray<double>().Select(d => (float)d).Select(func);
                X_test = X_raw.reshape(test_size, 1);
                Y_test = np.array(Y_raw);
            }
        }

        /// <summary>
        /// Construct the TF graphs that will be used for training and validation.
        /// In the former case, the optimizer is part of the graph, while it is not used for validation
        /// </summary>
        /// <returns>A TF Graph object.</returns>
        public override Graph BuildGraph()
        {
            const int n_input = 1;  // Number of features
            const int n_output = 1;  // Regression output is a number only

            var g = tf.get_default_graph();

            tf_with(tf.variable_scope("placeholder"), delegate
            {
                input = tf.placeholder(tf.float32, new TensorShape(-1, n_input));
                y = tf.placeholder(tf.float32, new TensorShape(-1));
            });

            tf_with(tf.variable_scope("FullyConnected"), delegate
            {
                var w1 = tf.get_variable("w1", shape: (n_input, n_hidden_layer_1), dtype: tf.float32, initializer: tf.random_normal_initializer(stddev: 1.0f));
                var b1 = tf.get_variable("b1", shape: n_hidden_layer_1, dtype: tf.float32, initializer: tf.random_normal_initializer(stddev: 1.0f));
                var z1 = tf.nn.relu(tf.matmul(input, w1) + b1);

                var w2 = tf.get_variable("w2", shape: (n_hidden_layer_1, n_hidden_layer_2), dtype: tf.float32, initializer: tf.random_normal_initializer(stddev: 1.0f));
                var b2 = tf.get_variable("b2", shape: n_hidden_layer_2, dtype: tf.float32, initializer: tf.random_normal_initializer(stddev: 1.0f));
                var z2 = tf.nn.relu(tf.matmul(z1, w2) + b2);

                var w3 = tf.get_variable("w3", shape: (n_hidden_layer_2, n_output), dtype: tf.float32, initializer: tf.random_normal_initializer(stddev: 1.0f));
                var b3 = tf.get_variable("b3", shape: n_output, dtype: tf.float32, initializer: tf.random_normal_initializer(stddev: 1.0f));
                y_hat = tf.matmul(z2, w3) + b3;
            });

            tf_with(tf.variable_scope("Loss"), delegate
            {
                loss_op = tf.reduce_mean(tf.square(tf.squeeze(y_hat) - y));
            });

            global_step = tf.Variable(1, trainable: false, name: "global_step");

            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, name: "train_op", global_step: global_step);

            return g;
        }

        public bool Run()
        {
            PrepareData();
            BuildGraph();
            Train();
            return true;
        }

        public override void Train()
        {
            var config = new ConfigProto
            {
                IntraOpParallelismThreads = 1,
                InterOpParallelismThreads = 1,
                LogDevicePlacement = true
            };

            using (var sess = tf.Session(config))
            {
                // init variables
                sess.run(tf.global_variables_initializer());

                var sw = new Stopwatch();
                sw.Start();

                // Check the loss before training starts.
                Console.WriteLine($"Initial validation loss: {((float)sess.run(loss_op, (input, X_test), (y, Y_test))):0}");

                // training
                foreach (var i in range(training_epochs))
                {
                    // by sampling some input data (fetching)
                    var (_, gs, loss) = sess.run((train_op, global_step, loss_op), (input, X_train), (y, Y_train));
                    //sess.run(train_op, (input, x_inputs_data), (y_true, y_inputs_data));

                    // We regularly check the loss
                    if (i > 0 && i % 1000 == 0)
                        Console.WriteLine($"iter:{i} - training loss: {((float)loss):0} validation loss: {((float)sess.run(loss_op, (input, X_test), (y, Y_test))):0}");
                }

                // Finally, we check our outgoing validation loss
                Console.WriteLine($"Final validation loss: {((float)sess.run(loss_op, (input, X_test), (y, Y_test))):0}");

                Console.WriteLine($"Training time: {sw.Elapsed.TotalSeconds:0}s");
            }

        }
    }
}
