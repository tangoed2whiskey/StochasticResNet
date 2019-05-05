import numpy as np 
import tensorflow as tf
from callbacks import step_decay, momentum_schedule, MomentumScheduler
from stochastic_activities import StochasticActivity

def main():
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    # x_train,y_train = x_train[:20000],y_train[:20000]

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(800, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.5),
    StochasticActivity(),
    tf.keras.layers.Dense(800, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.5),
    StochasticActivity(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    moment = MomentumScheduler(momentum_schedule)

    sgd = tf.keras.optimizers.SGD(clipnorm=15)
    model.compile(optimizer=sgd,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=100, callbacks=[lrate,moment])
    predictions = model.predict(x_test)
    correct = [True if np.argmax(pred)==true else False for pred,true in zip(predictions,y_test)]
    print('Number incorrect: {}'.format(len(correct)-np.sum(correct)))
    print('Accuracy on test: {:.3f}'.format(np.mean(correct)))



if __name__=='__main__':
    main()