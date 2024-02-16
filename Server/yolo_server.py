from threading import Thread
from ultralytics import YOLO
import numpy as np
import requests
import random
import socket
import json
import cv2
import os

HOST = socket.gethostbyname(socket.gethostname())
PORT = 8000
CLIENT_ORIGIN = 'chrome-extension://anfheiolininhjloaffbdnmhgnnjcnnh'
MIN_CONFIDENCE = 0.3

classes_dict = {'bicycles': [1],
                'cars': [2],
                'motorcycles': [3],
                'bus': [5], 'buses': [5],
                'traffic lights': [9],
                'a fire hydrant': [10], 'fire hydrants': [10],
                'vehicles': [1, 2, 3, 4, 5, 6, 7, 8]}
yolo8 = YOLO('YOLO-Weights/yolov8n.pt')


def initialize_server_socket():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f'Server listening on http://{HOST}:{PORT}')
    return server_socket


def handle_clients(server_socket):
    client_socket = None
    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f'Connected by <{client_address[0]}:{client_address[1]}>')
            client_socket.settimeout(10)
            thread = Thread(target=serve_client, args=[client_socket, client_address])
            thread.start()
        except socket.error as error:  # handles a failure in connecting to the client
            if client_socket is not None:
                client_socket.close()
            print('client failed,', error)


def serve_client(client_socket, client_address):
    while True:
        try:
            # Receive the request from the client
            request = client_socket.recv(4096)
        except TimeoutError:
            # Handle timeout error
            print(f'Connection with <{client_address[0]}:{client_address[1]}> timed out')
            client_socket.close()
            break

        # Check if the request is empty
        if not request:
            break

        # Decode the request
        request = request.decode()
        # Parse the request
        request_line = request.split('\r\n')[0]
        request_headers = '\r\n'.join(request.split('\r\n\r\n')[0].split('\r\n')[1:])
        request_content = json.loads(request.split('\r\n\r\n')[1])
        print(f'Request: {request_content}')
        challenge_type = request_content.get('challenge_type')
        requested_object = request_content.get('requested_object')
        payload_source = request_content.get('payload_source')
        # Check if the challenge type is an advanced iteration over special 3x3
        if challenge_type == 'special 3x3 - mini payloads':
            # Get list of the indexes corresponding to reCAPTCHA tiles that has the requested object
            tile_indexes = handle_mini_payloads(requested_object, payload_source, challenge_type)
            # Build a successful response to the client
            response = build_response(status_code=200, tile_indexes=tile_indexes)
        # Check if the requested object is supported
        elif requested_object in classes_dict.keys():
            # Download the payload
            payload_path = f'payloads/{random.randint(0, 10 ** 6)}.png'
            download_payload(payload_path, payload_source)
            # Get bounding boxes for the requested object
            bounding_boxes = get_bounding_boxes(payload_path, requested_object)
            # Get list of the indexes corresponding to reCAPTCHA tiles that has the requested object
            tile_indexes = get_tile_indexes(bounding_boxes, challenge_type)
            # Delete payload from server
            os.remove(payload_path)
            # Build a successful response to the client
            response = build_response(status_code=200, tile_indexes=tile_indexes)
        else:
            # Build a failure response to the client
            response = build_response(status_code=400, requested_object=requested_object)

        # Send the response to the client
        client_socket.sendall(response)


# Handles further iterations over special 3x3
# Each mini payload is the tile's image instead of the entire grid's
def handle_mini_payloads(requested_object, mini_payloads, challenge_type):
    # Merge the mini payloads into a big payload
    merged_payload_path = merge_mini_payloads(mini_payloads)
    # Get bounding boxes for the requested object
    bounding_boxes = get_bounding_boxes(merged_payload_path, requested_object)
    # Get list of the indexes corresponding to reCAPTCHA tiles that has the requested object
    tile_indexes = get_tile_indexes(bounding_boxes, challenge_type)
    # Delete payload from server
    os.remove(merged_payload_path)
    return tile_indexes


# Merges the mini payloads into a big payload based on their position in the grid
# Returns the path to the merged payload
def merge_mini_payloads(mini_payloads):
    mini_payloads_dict = {}
    indexes = mini_payloads.get('indexes')
    sources = mini_payloads.get('sources')
    for i in range(len(indexes)):
        index = indexes[i]
        source = sources[i]
        row, col = index // 3, index % 3
        mini_payload_path = f'payloads/mini/{random.randint(0, 10 ** 6)}.png'
        download_payload(mini_payload_path, source)
        mini_payloads_dict[(row, col)] = mini_payload_path

    # initialize the merged payload as a black image
    merged_payload = np.zeros((300, 300, 3))

    # go over all the possible tile coordinates in the image
    for i in range(3):
        for j in range(3):
            # checks if the current coordinate belongs to a mini payload
            if (i, j) in mini_payloads_dict.keys():
                # read the mini payload's stream from the saved file
                mini_payload = cv2.imread(mini_payloads_dict.get((i, j)))
                # set the merged payload's data at the current tile to the mini payload's content
                merged_payload[i * 100: (i + 1) * 100, j * 100: (j + 1) * 100] = mini_payload[0: 100, 0: 100]

    # set the merged payload's path
    merged_payload_path = f'payloads/merged{random.randint(0, 10 ** 6)}.png'
    # save the merged payload to the sever
    cv2.imwrite(merged_payload_path, merged_payload)

    # iterate over all mini payloads to delete their files from the server
    for mini_payload_path in mini_payloads_dict.values():
        os.remove(mini_payload_path)

    return merged_payload_path


def download_payload(payload_path, payload_source):
    try:
        response = requests.get(payload_source)
        if response.status_code == 200:
            f = open(payload_path, 'wb')
            f.write(response.content)
            f.close()
    except requests.exceptions.RequestException as e:
        print(f'An error occurred while downloading payload: {e}')


def get_bounding_boxes(payload_path, requested_object):
    bounding_boxes = list()
    # Find the COCO-dataset index of the requested object
    requested_class_indexes = classes_dict.get(requested_object)
    # Generate detections
    results = yolo8(payload_path)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            predicted_class_index = int(box.cls[0])
            if predicted_class_index in requested_class_indexes:
                # Find confidence of detection
                conf = round(float(box.conf[0]), 2)
                # Filter out low-confidence detections
                if conf >= MIN_CONFIDENCE:
                    bounding_box = tuple(int(param) for param in box.xyxy[0])
                    bounding_boxes.append(bounding_box)

    return bounding_boxes


def get_tile_indexes(bounding_boxes, challenge_type):
    tile_indexes = list()
    # The order of the tiles matrix
    matrix_order = 3 if '3' in challenge_type else 4
    # The size (in pixels) of the payload
    payload_size = 300 if '3' in challenge_type else 450
    # The size of each tile (in pixels)
    tile_size = payload_size / matrix_order
    # Find for each tile if it needs to be selected
    for i in range(matrix_order):  # rows
        for j in range(matrix_order):  # cols
            tile_x1, tile_x2 = j * tile_size, (j + 1) * tile_size
            tile_y1, tile_y2 = i * tile_size, (i + 1) * tile_size
            for bounding_box in bounding_boxes:
                box_x1, box_y1, box_x2, box_y2 = bounding_box
                if box_x1 < tile_x2 and box_x2 > tile_x1 and box_y1 < tile_y2 and box_y2 > tile_y1:
                    tile_indexes.append(str(i * matrix_order + j))
                    break

    return tile_indexes


def build_response(status_code, tile_indexes=None, requested_object=None):
    response_line = None
    response_headers = f'Content-Type: text/plain\r\nAccess-Control-Allow-Origin: {CLIENT_ORIGIN}'
    response_content = None

    if status_code == 200:
        response_line = 'HTTP/1.1 200 OK'
        response_content = ' '.join(tile_indexes)
    elif status_code == 400:
        response_line = 'HTTP/1.1 400 Bad Request'
        response_content = f'There is no support for {requested_object} yet. refresh the reCAPTCHA and try again'

    print(f'Response: {response_content}')
    response = f'{response_line}\r\n{response_headers}\r\n\r\n{response_content}'.encode()
    return response


def main():
    try:
        server_socket = initialize_server_socket()
        handle_clients(server_socket)
        server_socket.close()
    except socket.error as error:  # handles a failure in initializing the server_socket
        print(f'Connection Failure: {error}\n' + 'Terminating program...')


if __name__ == '__main__':
    main()
