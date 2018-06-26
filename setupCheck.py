import tensorflow as tf
import getpass

a = tf.get_variable(name="a", initializer = tf.constant(5))
b = tf.Variable(7)

c = tf.add(a,b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("5 + 7 = " + str(sess.run(c)))

if sess.run(c) == 12:
	print("Your are ready, " + str(getpass.getuser()))


