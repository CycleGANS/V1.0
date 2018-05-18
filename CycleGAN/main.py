import tensorflow as tf
import train
from test import test

if __name__ == '__main__':
    train.training(dataset='monet2photo', epochs=75, image_shape=256, batch_size=1, G_cyc_loss_lambda=10.0, F_cyc_loss_lambda=10.0, learning_rate=0.0002)
    print("Training completed!")
    test(dataset_str='monet2photo', img_width=256, img_height=256)
    print("Testing completed! Enjoy your life!!!")
