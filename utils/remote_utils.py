from paramiko import SSHClient
from scp import SCPClient
import sys
import os
import xml.etree.ElementTree as ET
import tempfile
import uuid

REMOTE_HOSTS = []

class Remote:
    def __init__(self, host):
        self.host = host
    
    def open(self):
        return self.__enter__()
    
    def close(self):
        return self.__exit__()

    def __enter__(self):
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect(self.host)

        # you can also use progress4, which adds a 4th parameter to track IP and port
        # useful with multiple threads to track source
        def progress4(filename, size, sent, peername):
            sys.stdout.write("(%s:%s) %s's progress: %.2f%%   \r" % (peername[0], peername[1], filename, float(sent)/float(size)*100) )
        self.scp = SCPClient(self.ssh.get_transport(), progress4=progress4)

        return self

    def __exit__(self, *args):
        self.scp.close()
        self.ssh.close()

    def get(self, remote_path, local_path=".", **kwargs):
        self.scp.get(remote_path, local_path, **kwargs)
    
    def put(self, files, remote_path, **kwargs):
        self.scp.put(files, remote_path, **kwargs)

    def read_file(self, remote_path, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            fpath = os.path.join(tmpdirname, str(uuid.uuid4()))
            self.get(remote_path, fpath)
            with open(fpath) as f:
                contents = f.read()
        return contents


    def execute_commands(self, commands, return_by_lines=True):
        """
        Execute multiple commands in succession.

        :param List[str] commands: List of unix commands as strings.
        """
        for cmd in commands:
            stdin, stdout, stderr = self.ssh.exec_command(cmd)
            stdout.channel.recv_exit_status()
            if not return_by_lines:
                yield stdout.read()
                continue
            response = stdout.readlines()
            print(stderr.read())
            for line in response:
                print(
                    f"INPUT: {cmd}\n \
                    REMOTE: {line}"
                )
                yield line

    def get_by_pattern(self, remote_root_dir, pattern, local_dir):
        cmds = [f"ls {remote_root_dir} | grep {pattern}"]
        output = list(self.execute_commands(cmds))
        if len(output) == 0:
            return False
        fname = output[0].rstrip('\n')
        remote_fpath = os.path.join(remote_root_dir, fname)
        local_fpath = os.path.join(local_dir, fname)
        print(f"Fetching {remote_fpath} --> {local_fpath}")
        self.get(remote_fpath, local_dir)
        return True

    def get_users_by_pids(self, pids):
        cmds = [f"ps -o pid= -o ruser= -p {','.join(pids)}"]
        output = self.execute_commands(cmds)
        pid2user = {}
        for line in output:
            pid, user = line.split()
            pid2user[pid] = user
        return pid2user

    def get_state(self, gpu_infos):
        num_occupied = 0
        for gpu in gpu_infos:
            if len(gpu["processes"]) > 0:
                num_occupied += 1

        state = None
        if num_occupied == len(gpu_infos):
            state = "red"
        elif num_occupied > 0:
            state = "yellow"
        else:
            state = "green"
        return state


    def get_gpu_info(self):
        cmds = ["nvidia-smi -q -x"]
        response = list(self.execute_commands(cmds, False))[0]
        # response = response.decode("utf-8")
        # import pdb; pdb.set_trace()
        try:
            root = ET.fromstring(response)
        except:
            import pdb; pdb.set_trace()
        
        gpus = root.findall('gpu')

        gpu_infos = []
        
        for idx, gpu in enumerate(gpus):
            model = gpu.find('product_name').text
            procs = gpu.findall('processes')[0]

            processes_info = [
                dict(
                    pid=process.find('pid').text,
                    process_name=process.find('process_name').text,
                    used_memory=process.find('used_memory').text
                )
                for process in procs]
            
            pids = [p['pid'] for p in processes_info]
            if len(pids):
                pid2user = self.get_users_by_pids(pids)
                for p in processes_info:
                    p['user'] = pid2user[p['pid']]

            # utilization
            utilization = gpu.find('utilization')
            gpu_util = utilization.find('gpu_util').text
            mem_util = utilization.find('memory_util').text
            util = dict(
                gpu_util=gpu_util,
                memory_util=mem_util
            )
            
            gpu_infos.append({'id': idx, 'model': model, 'processes': processes_info, 'utilization': util})

        return gpu_infos

    def reverse_forward_port(self, local_port, remote_port):
        command = f"ssh -f -N -T -R {remote_port}:localhost:{local_port} {self.host}"
        os.system(command)
        



def get_free_hosts(hosts=REMOTE_HOSTS):
    free_hosts = []
    for host in REMOTE_HOSTS:
        with Remote(host) as remote:
            if remote.get_state(remote.get_gpu_info()) == "green":
                free_hosts.append(host)
    return free_hosts


def print_xml_element(element):
    print(ET.tostring(element, encoding='utf8').decode('utf8'))