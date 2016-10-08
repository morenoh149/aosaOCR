import json


def do_POST(s):
    response_code = 200
    response = ""
    var_len = int(s.headers.get('Content-length'))
    content = s.rfile.read(var_len)
    payload = json.loads(content)

    if payload.get('train'):
        nn.train(payload['trainArray'])
        nn.save()
    elif payload.get('predict'):
        try:
            response = {
                "type": "test",
                "result": nn.predict(str(payload['image']))
            }
        except:
            response_code = 500
    else:
        response_code = 400

    s.send_response(response_code)
    s.send_header("Content-type", "application/json")
    s.send_header("Access-Control-Allow-Origin", "*")
    s.end_headers()
    if response:
        s.wfile.write(json.dumps(response))
    return


def save(self):
    if not self._use_file:
        return

    json_neural_network = {
        "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
        "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
        "b1": self.input_layer_bias[0].tolist()[0],
        "b2": self.hidden_layer_bias[0].tolist()[0]
    }
    with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
        json.dump(json_neural_network, nnFile)


def _load(self):
    if not self._use_file:
        return

    with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
        nn = json.load(nnFile)
    self.theta1 = [np.array(li) for li in nn['theta1']]
    self.theta2 = [np.array(li) for li in nn['theta2']]
    self.input_layer_bias = [np.array(nn['b1'][0])]
    self.hidden_layer_bias = [np.array(nn['b2'][0])]
