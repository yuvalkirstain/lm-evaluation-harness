import subprocess


def main():
    start = 42123
    for i in range(start + 1000):
        process = subprocess.Popen(["scancel",
                                    str(i + start)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print("output:")
        print(stdout.decode("utf-8"))
        print("err:")
        print(stderr.decode("utf-8"))


if __name__ == '__main__':
    main()
