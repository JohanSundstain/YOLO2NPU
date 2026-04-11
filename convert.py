import argparse
import configparser
import subprocess

def arg_parse():
	parser = argparse.ArgumentParser(description="Запуск конвертации")
	parser.add_argument("-c", "--conf",type=str, help="Путь до файла конфигурации")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = arg_parse()
	config_file = args.conf

	command = ["cixbuild", config_file]
		
	result = subprocess.run(command) 