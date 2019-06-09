import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        y = nn.as_scalar(self.run(x))
        if y < 0:
            return -1
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        update = True
        while update:
            update = False
            for x, y in dataset.iterate_once(batch_size):
                true_y = nn.as_scalar(y)
                predict_y = self.get_prediction(x)
                if true_y != predict_y:
                    update = True
                    self.w.update(x, true_y)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer = 100                          # size of the hidden layer
        self.w1 = nn.Parameter(1, hidden_layer)     # weights for layer 1
        self.w2 = nn.Parameter(hidden_layer, 1)     # weights for layer 2
        self.b1 = nn.Parameter(1, hidden_layer)     # bias for layer 1
        self.b2 = nn.Parameter(1, 1)                # bias for layer 2
        self.learning_rate = -0.03                  # learning rate

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        batch = nn.Linear(x, self.w1)
        predicted_y = nn.AddBias(batch, self.b1)
        relu = nn.ReLU(predicted_y)
        batch = nn.Linear(relu, self.w2)
        predicted_y = nn.AddBias(batch, self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 10
        counter = 500
        update = True
        while update:
            total_loss = 0
            total = 0
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                total_loss += nn.as_scalar(loss)
                total += 1
                grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad_w1, self.learning_rate)
                self.w2.update(grad_w2, self.learning_rate)
                self.b1.update(grad_b1, self.learning_rate)
                self.b2.update(grad_b2, self.learning_rate)
            avg_loss = total_loss / total
            if avg_loss < 0.02:
                update = False
    
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        OUTPUT = 10                 # size of the output and hidden layer for the 3rd layer
        HIDDEN_LAYER1 = 80          # sife of the hidden layer for the 1st layer
        HIDDEN_LAYER2 = 80          # size of the hidden layer for the 2nd layer
        self.w1 = nn.Parameter(784, HIDDEN_LAYER1)
        self.b1 = nn.Parameter(1, HIDDEN_LAYER1)
        self.w2 = nn.Parameter(HIDDEN_LAYER1 , HIDDEN_LAYER2)
        self.b2 = nn.Parameter(1, HIDDEN_LAYER2)
        self.w3 = nn.Parameter(HIDDEN_LAYER2, OUTPUT)
        self.b3 = nn.Parameter(1, OUTPUT)
        self.learning_rate = -0.03

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        data1 = nn.Linear(x, self.w1)
        predicted_y1 = nn.AddBias(data1, self.b1)
        relu1 = nn.ReLU(predicted_y1)

        data2 = nn.Linear(relu1, self.w2)
        predicted_y2 = nn.AddBias(data2, self.b2)
        relu2 = nn.ReLU(predicted_y2)

        data3 = nn.Linear(relu2, self.w3)
        predicted_y3 = nn.AddBias(data3, self.b3)

        return predicted_y3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model
        """
        "*** YOUR CODE HERE ***"
        bacth_size = 20
        accuracy = 0
        check = False       # boolean counter for checking the accuracy for each batch, not the whole data set
        while True:
            for x, y in dataset.iterate_once(bacth_size):
                loss = self.get_loss(x, y)
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(loss, \
                    [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grad_w1, self.learning_rate)
                self.w2.update(grad_w2, self.learning_rate)
                self.w3.update(grad_w3, self.learning_rate)
                self.b1.update(grad_b1, self.learning_rate)
                self.b2.update(grad_b2, self.learning_rate)
                self.b3.update(grad_b3, self.learning_rate)

                # return once we get the desired accuracy
                if check and dataset.get_validation_accuracy()> 0.974:
                    return
            # check the accuracy of each batch if our accuracy is almost 97 %
            if dataset.get_validation_accuracy() > 0.968:
                check = True
                
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_size = 100
        output = 5
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.learning_rate = -0.03
        self.w1 = nn.Parameter(self.num_chars, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        self.w2 = nn.Parameter(hidden_size, hidden_size)
        self.b2 = nn.Parameter(1, output)
        self.w3 = nn.Parameter(hidden_size, output)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        initial = True
        for words in xs:
            if initial:
                linearize = nn.Linear(words, self.w1)
                w_bias = nn.AddBias(linearize, self.b1)
                z = nn.ReLU(w_bias)
                initial = False
            else:
                linearize = nn.Add(nn.Linear(words, self.w1), nn.Linear(z, self.w2))
                w_bias = nn.AddBias(linearize, self.b1)
                z = nn.ReLU(w_bias)
        linearize = nn.Linear(z, self.w3)
        output = nn.AddBias(linearize, self.b2)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 10
        check = False
        while True:
            for xs, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(xs ,y)
                grad_w1, grad_w2, grad_w3, grad_b1, grad_b2 = nn.gradients(loss, [self.w1, \
                    self.w2, self.w3, self.b1, self.b2])
                self.w1.update(grad_w1, self.learning_rate)
                self.w2.update(grad_w2, self.learning_rate)
                self.w3.update(grad_w3, self.learning_rate)
                self.b1.update(grad_b1, self.learning_rate)
                self.b2.update(grad_b2, self.learning_rate)
                if check and dataset.get_validation_accuracy() > 0.84:
                    return
            if dataset.get_validation_accuracy() > 0.81:
                check = True



