# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code


#El rago de valores de entrenamiento son el 70% del total
rango_entrenamiento_x= len(x_data)* 0.7
rango_entrenamiento_y= len(y_data)* 0.7

#Con el floor acotamos hacia arriba (floor) y pillamos todos los valores desde el primero (:) hasta el rango de entrenamiento acotado hacia abajo con un casteo
datos_entrenamiento_x=x_data[:int(np.floor(rango_entrenamiento_x))]
datos_entrenamiento_y=y_data[:int(np.floor(rango_entrenamiento_y))] #Como limite el rango de entrnamiento

#El rango de validacion es el rango de entrenamiento sumando un 15% mas del total
rango_validacion_x= rango_entrenamiento_x + len(x_data)*0.15
rango_validacion_y= rango_entrenamiento_y + len(y_data)*0.15

#Lo datos de validacion seran pues los que esten comprendidos entre el rango de entrenamiento y el rango de validacion y con el round redondeamos
datos_validacion_x= x_data[int(np.round(rango_entrenamiento_x)):int(np.round(rango_validacion_x))]
datos_validacion_y= y_data[int(np.round(rango_entrenamiento_y)):int(np.round(rango_validacion_y))]

#Los datos de test son los comprendidos entre los de validacion y el final (usamos el redondeo otra vez)
datos_test_x= x_data[int(np.round(rango_validacion_x)):]
datos_test_y= y_data[int(np.round(rango_validacion_y)):]


#Lista para almacenar los errores y mostrarlos en grafica
lista_entrenamiento= []
lista_validacion=[]

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
#Epoch=epoca, para ir viendo el error a medida que se ejecuta el programa
epoch=0
diferencia=100.0 #Para calcular la diferencia entre el error actual y el anterior
while diferencia>0.001:
    epoch+=1
    for jj in xrange(len(datos_entrenamiento_x) / batch_size):
        batch_xs = datos_entrenamiento_x[jj * batch_size: jj * batch_size + batch_size] #Los cogemos de 20 en 20 y llamamos a la funcion train pasandoselos por parametro
        batch_ys = datos_entrenamiento_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    #Errores entrenamiento

    #Llamamos a la funcion loss para el calculo del error en el entrenamiento y añadimos ese valor luego nuestra lista de entrenamiento
    dato_entrenamiento= sess.run(loss,feed_dict={x: batch_xs, y_: batch_ys})
    lista_entrenamiento.append(dato_entrenamiento)
    print "Epoca Entrenamiento:",epoch,"Error:",dato_entrenamiento #Imprimimos en que epica estamos y que error tiene en esa epoca
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(result, batch_ys): #Mediante el zip vamos viendo tanto el result (el resultado de la neurona como y que seria la etiqueta)
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

    #Errores Validacion

    #Llamamos a la funcion loss tambien pero ahora para el calculo del error en los datos de validacion, en vez de los datos de entrenamiento. (se los pasamos por parametro)
    dato_validacion= sess.run(loss,feed_dict={x: datos_validacion_x, y_: datos_validacion_y})
    lista_validacion.append(dato_validacion) #Unavez calculado el error en la validacion se lo pasamos a la lista
    print "Epoca Validacion:", epoch, "Error:", dato_validacion #Mostramos la epoca y el error en la validacion

    if epoch>1: #Tiene minimo que haber 2 valores para poder hacer una resta por eso empezamos en mayor que 1
        diferencia= abs(lista_validacion[-2] - dato_validacion) #Cogemos el error anterior y le restamos el error actual
    print "Diferencia",diferencia #Imprimimos la diferencia entre ambos
    print "----------------------------------------------------------------------------------"

#print lista_entrenamiento
#print lista_validacion

# Errores de validacion 15%
print "----------------------"
print "   Test result...     "
print "----------------------"

total = 0.0
error = 0.0
datos_test = sess.run(y, feed_dict={x: datos_test_x}) #Llamamos a la funcion pasandole los datos de test
for b, r in zip(datos_test_y, datos_test): #recorriendo tanto la etiqueta como el resultado vamos comprobando el maximo en ambos, si no son iguales sumamos 1 a un contador de errores
    if np.argmax(b) != np.argmax(r):
        error += 1
    total += 1 # En cada iteracion sumamos 1 al total para comprobar cuantas iteraciones hay y asi poder calcular el porcentaje de error y de acierto
fail = error / total * 100.0 #Calculo del porcentaje de error
print "Porcentaje de error: ", fail,"% y portenjate de exito", (100.0 - fail), "%" #Imprimimos por consola tanto el porcentaje de error como el de acierto, este ultimo es el total menos el del error

plot.ylabel('Errores') #Eje y se llamara Errores
plot.xlabel('Epocas') #Eje x se llamara Epocas
entrada_entrenamiento, = plot.plot(lista_entrenamiento) #Creamos una grafica pasandole la lista de entrenamiento (al pasar 1 parametro toma como eje x valores de 1..N-1
entrada_validacion, = plot.plot(lista_validacion) #Igual que en el anterior
plot.legend(handles=[entrada_entrenamiento, entrada_validacion], #Creamos una grafica donde aparezcan las 2 anteriores
            labels=['Error entrenamiento', 'Error validacion']) #Añadimos una legenda para mejor entendimiento
plot.savefig('Grafica_entrenamiento_validacion.png') #La guardamos en la carpeta del proyecto




