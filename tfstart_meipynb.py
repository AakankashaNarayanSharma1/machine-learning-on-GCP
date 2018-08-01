import sys
sys.path.insert(0,'F:/tf/Lib/site-packages')
# coding: utf-8
# <h2> Solution to Halley's method for finding root's of a fourth degree polynomial </h2>
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/142614c0378a1d61cb623c1352bf85b6b7bc4397" />
import tensorflow as tf
f = lambda a, x0 : a[0] + a[1]*x0 + a[2] * (x0**2) + a[3] * (x0**3) + a[4] * (x0**4)
f1 = lambda a, x0 : a[1] + 2 * a[2] * x0 + 3 * a[3] * (x0**2) + 4 * a[4] * (x0**3)
f2 = lambda a, x0 : 2 * a[2] + 6 * a[3] * x0 + 12 * a[4] * (x0**2)

def halley_s_method(a):
    x0=a[5]
    x1 = x0 - ((2*f(a,x0)*f1(a,x0))/(2*f1(a,x0)*f1(a,x0) - f(a,x0)*f2(a,x0)))
    cond = lambda x1, x0 : ((x1 - x0) > 0.05)             ## will change for better accuracy
    body = lambda x1, x0 : (x1 - ((2*f(a,x0)*f1(a,x0))/(2*f1(a,x0)*f1(a,x0) - f(a,x0)*f2(a,x0))), x1)
    final=tf.while_loop(cond, body, [x1,x0])
    return x1

## variables for coefficients
a = tf.placeholder(shape=(6,), dtype=tf.float32)
# x0 = tf.placeholder(shape=(), dtype=tf.float32)

# a = tf.constant([-4.,1.,1.,1.,1.,1.])
sess=tf.Session()
res=halley_s_method(a)
print(sess.run(res, feed_dict={a:[-4.,1.,1.,1.,1.,1.]}))
# print(sess.run(res))
# print(sol)
sess.close()