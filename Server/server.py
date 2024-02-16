from keras.models import load_model
from threading import Thread
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

classes_dict = {'bicycles': [0],
                'bridges': [1],
                'bus': [2], 'buses': [2],
                'cars': [3],
                'crosswalks': [4],
                'a fire hydrant': [5], 'fire hydrants': [5],
                'motorcycles': [6],
                'palm trees': [7],
                'stairs': [8],
                'traffic lights': [9],
                'vehicles': [0, 2, 3, 6]}
model = load_model('models/top_model.h5')


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
        # Check if the challenge is supported
        if challenge_type != '4x4' and requested_object in classes_dict.keys():
            # Check if the challenge type is an advanced iteration over special 3x3
            if challenge_type == 'special 3x3 - mini payloads':
                # Get a dictionary in the form of {'indexes': [indexes_here], 'paths': [paths_here]}
                mini_payloads_dict = download_mini_payloads(payload_source)
            else:
                # Download the 3x3 payload
                payload_path = f'payloads/regular/{random.randint(0, 10 ** 6)}.png'
                download_payload(payload_path, payload_source)
                # Split the 3x3 payload to 9 mini payloads
                # Result is a dictionary of the form {'indexes': [0, ..., 9], 'paths': [path1, ..., path9]}
                mini_payloads_dict = split_payload(payload_path)
                # delete the 3x3 payload from the server as it has already served its purpose
                os.remove(payload_path)

            # Get list of the indexes corresponding to reCAPTCHA tiles that has the requested object
            tile_indexes = get_tile_indexes(requested_object, mini_payloads_dict)
            # Delete the mini-payloads from the server as they have already served their purpose
            delete_mini_payloads(mini_payloads_dict.get('paths'))
            # Build a successful response to the client
            response = build_response(status_code=200, tile_indexes=tile_indexes)
        else:
            # Build a failure response to the client
            response = build_response(status_code=400, requested_object=requested_object)

        # Send the response to the client
        client_socket.sendall(response)


# Gets a dictionary of mini-payload indexes and source
# Downloads each mini-payload by source and save to a path
# Returns a dictionary of mini-payload indexes and paths
def download_mini_payloads(mini_payload_sources):
    indexes = mini_payload_sources.get('indexes')
    sources = mini_payload_sources.get('sources')
    mini_payloads_dict = {'indexes': [index for index in indexes], 'paths': []}
    # iterate over each source, download it, and add its path to mini_payloads_dict
    for mini_payload_source in sources:
        mini_payload_path = f'payloads/mini/{random.randint(0, 10 ** 6)}.png'
        download_payload(mini_payload_path, mini_payload_source)
        mini_payloads_dict['paths'].append(mini_payload_path)

    return mini_payloads_dict


def download_payload(payload_path, payload_source):
    try:
        response = requests.get(payload_source)
        if response.status_code == 200:
            f = open(payload_path, 'wb')
            f.write(response.content)
            f.close()
    except requests.exceptions.RequestException as e:
        print(f'An error occurred while downloading payload: {e}')


def split_payload(payload_path):
    # initializes the dict to a list of all tile indexes and an empty list of mini-payload paths
    mini_payloads_dict = {'indexes': [index for index in range(10)], 'paths': []}
    # read the 3x3 payload
    payload = cv2.imread(payload_path)
    # iterate over it and each time extract one mini payload
    for index in range(8):
        row = index // 3
        col = index % 3
        mini_payload_path = f'payloads/mini/{random.randint(0, 10 ** 6)}.png'
        # get the data from the region of the current mini payload's
        img = payload[row * 100: (row + 1) * 100, col * 100: (col + 1) * 100]
        cv2.imwrite(mini_payload_path, img)
        mini_payloads_dict['paths'].append(mini_payload_path)

    return mini_payloads_dict


# Filters out the indexes that belong to mini-payloads who don't have the requested object
def get_tile_indexes(requested_object, mini_payloads_dict):
    tile_indexes = []
    mini_payload_indexes = mini_payloads_dict.get('indexes')
    mini_payload_paths = mini_payloads_dict.get('paths')
    # send the paths to the model to determine for each one if the payload is an image of the requested object
    # results is a list the same size as paths consisting of a boolean value for each mini-payload
    results = is_requested_object(mini_payload_paths, requested_object)
    for i, result in enumerate(results):
        if result:
            tile_indexes.append(str(mini_payload_indexes[i]))

    return tile_indexes


def is_requested_object(mini_payload_paths, requested_object):
    results = []
    X = []
    for mini_payload_path in mini_payload_paths:
        mini_payload = cv2.imread(mini_payload_path)
        normalized_mini_payload = mini_payload / 255
        X.append(normalized_mini_payload)

    X = np.array(X)
    requested_class_indexes = classes_dict.get(requested_object)

    yhat = model.predict(X)
    yhat = np.argmax(yhat, axis=1)

    for i in range(len(X)):
        results.append(yhat[i] in requested_class_indexes)

    return results


def delete_mini_payloads(mini_payload_paths):
    for path in mini_payload_paths:
        os.remove(path)


def build_response(status_code, tile_indexes=None, requested_object=None):
    response_line = None
    response_headers = f'Content-Type: text/plain\r\nAccess-Control-Allow-Origin: {CLIENT_ORIGIN}'
    response_content = None

    if status_code == 200:
        response_line = 'HTTP/1.1 200 OK'
        response_content = ' '.join(tile_indexes)
    elif status_code == 400:
        response_line = 'HTTP/1.1 400 Bad Request'
        if requested_object not in classes_dict.keys():
            response_content = f'There is no support for {requested_object} yet. refresh the reCAPTCHA and try again'
        else:
            response_content = f'There is no support for 4x4 challenges yet. refresh the reCAPTCHA and try again'

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
