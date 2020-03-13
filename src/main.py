import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def main():
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))

if __name__ == "__main__":
    main()
