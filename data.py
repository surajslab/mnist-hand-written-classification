import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy
 
mnist =keras.datasets.mnist
(x_train,y_train) , (x_test,y_test) = mnist.load_data()

x_train=x_train/255.0
x_test=x_test/255.0

#model defining
model=keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

#model compiling
model.compile(

    optimizer='adam',
    loss ='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=3)
loss,accu =model.evaluate(x_test,y_test)
print("loss :"+str(loss) +" "+"accuracy :" +str(accu))

#saving the model
model.save("number_reader.model")

#loading the saved model
new_model=keras.models.load_model("number_reader.model")

#predicting using the model
prediction=model.predict([x_test])

print(numpy.argmax(prediction[0]))
plt.imshow(x_test[0])
plt.show()






