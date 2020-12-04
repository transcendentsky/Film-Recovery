import argparse
# Train Parameters
parser = argparse.ArgumentParser(description='Unwarp Film Train Configure')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 40)')  # 50 for 4 gpu
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',  # 100 for 4 gpu
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.85, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_intervals', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--visualize_para', action='store_true', default=False,
                    help='For visualizing the Model parameters')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Load model parameters from pretrained model')
parser.add_argument('--pretrained_model_dir', type=str, default='model/tv_constrain_3.pkl',
                    help='load pretrained model path')
parser.add_argument('--train_path', type=str, default='/home1/qiyuanwang/film_generate/npy/', help='train dataset path')
parser.add_argument('--test_path', type=str, default='/home1/qiyuanwang/film_generate/npy_test/', help='test dataset path')

parser.add_argument('--use_mse', type=bool, default=False, help='use mse loss')

# Save path and output
parser.add_argument('--save_model_dir', type=str, default='/home1/quanquan/film_code/model2/', help='Save the model in this path')
parser.add_argument('--output_dir', type=str, default='/home1/quanquan/film_code/test_output2/', help='output image')
parser.add_argument('--write_summary', type=bool, default=True, help='write tensor board')
parser.add_argument('--write_txt', type=bool, default=True, help='write txt')
parser.add_argument('--write_image_train', type=bool, default=True, help='write train image')
parser.add_argument('--write_image_val', type=bool, default=False, help='write val image')
parser.add_argument('--write_image_test', type=bool, default=True, help='write test image')
parser.add_argument('--calculate_CC', type=bool, default=True, help='calculate cc')
parser.add_argument('--model_name', type=str, default='tv_constrain', help='constrain')
args = parser.parse_args()
