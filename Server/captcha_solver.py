from keras.models import load_model
import numpy as np
import cv2
import os


model = load_model('models/top_model.h5')
class_names = os.listdir('Dataset')


def solve_captcha(captcha_path, requested_object):
    payload = cv2.imread(captcha_path)
    mini_payloads = []
    for row in range(3):
        for col in range(3):
            mini_payload = payload[row * 100: (row + 1) * 100, col * 100: (col + 1) * 100]
            mini_payloads.append(mini_payload)

    mini_payloads = np.array(mini_payloads) / 255
    output = model.predict(mini_payloads)
    # name of the object in each tile
    objects = [class_names[i] for i in np.argmax(output, axis=1)]
    # array of booleans representing presence of requested object in every tile
    solution = [obj == requested_object for obj in objects]

    return solution


def mark_solution(captcha_path, requested_object, solution):
    check_mark = cv2.imread('images/Check mark.png')
    cross_mark = cv2.imread('images/Cross mark.png')
    payload = cv2.imread(captcha_path)
    # mark presence for each tile
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            # check if the object in the tile is the requested object
            if solution[index]:
                height, width = check_mark.shape[: 2]
                payload[row * 100: row * 100 + height, col * 100: col * 100 + width] = check_mark[:, :]
            else:
                height, width = cross_mark.shape[: 2]
                payload[row * 100: row * 100 + height, col * 100: col * 100 + width] = cross_mark[:, :]

    # add caption space
    temp = np.ones((350, 300, 3)) * 255
    temp[50: 350, 0: 300] = payload[:, :]
    payload = temp

    # write caption
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 0, 35
    spaces = ''.join([' ' for i in range((18 - len(requested_object)) // 2)])

    x, y = (18 - len(requested_object)) // 2, 35
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2

    cv2.putText(payload, spaces + requested_object, (x, y), font, fontScale, color, thickness)

    solution_path = f"payloads/solved/{captcha_path.split('/')[-1]}"
    cv2.imwrite(solution_path, payload)

    return solution_path


def main():
    captcha_path = input('Enter the captcha path: ')
    print(f'Object names: {class_names}')
    requested_object = input('Enter the requested object: ')
    solution = solve_captcha(captcha_path, requested_object)
    solution_image_path = mark_solution(captcha_path, requested_object, solution)
    solution_image = cv2.imread(solution_image_path)
    cv2.imshow('Solution', solution_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
