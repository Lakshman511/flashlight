import numpy as np
from pydub import AudioSegment
import socket
import argparse

def get_float_arry(sound):
    sound = sound.set_frame_rate(16000)
    channel_sounds = sound.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", help="IP address of the server", type=str)
    parser.add_argument("--file_path", help="Path of the audio file to be sent", type=str)
    args = parser.parse_args()
    sound = AudioSegment.from_file(args.file_path,args.file_path[args.file_path.rfind('.')+1:])
    fp_arr = get_float_arry(sound)#[:,0]
    print(fp_arr.shape)
    print("sum = ",sum(fp_arr))
    print(*fp_arr[1980:2000])
    fp_arr = np.insert(fp_arr, 0, len(fp_arr))
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.server_ip,8888))
    client.send(bytes(fp_arr))
    from_server = client.recv(4096)
    client.close()
    print(from_server)
main()