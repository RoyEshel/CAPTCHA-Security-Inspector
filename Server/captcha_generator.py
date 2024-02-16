import numpy as np
import random
import cv2
import os


dataset_path = 'Test'


def generate_captchas(captchas_amount):
    generated_captcha_paths = []
    class_names = os.listdir(dataset_path)
    for i in range(captchas_amount):
        generated_captcha_path = f'payloads/generated/{random.randint(0, 10 ** 6)}.png'
        generated_captcha = np.empty((300, 300, 3))
        for j in range(9):
            row, col = j // 3, j % 3
            random_class_name = random.choice(class_names)
            class_path = os.path.join(dataset_path, random_class_name)
            image_names = os.listdir(class_path)
            random_image_name = random.choice(image_names)
            image_path = os.path.join(class_path, random_image_name)
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (100, 100))
            generated_captcha[row * 100: (row + 1) * 100, col * 100: (col + 1) * 100] = resized_image[0: 100, 0: 100]

        cv2.imwrite(generated_captcha_path, generated_captcha)
        generated_captcha_paths.append(generated_captcha_path)

    return generated_captcha_paths


def main():
    captchas_amount = int(input('Enter Amount of Captchas: '))
    generated_captcha_paths = generate_captchas(captchas_amount)
    for i, generated_captcha_path in enumerate(generated_captcha_paths):
        print(f'{i + 1}) {generated_captcha_path}')


if __name__ == '__main__':
    main()
