def read_version() -> str:
    with open("version.txt", "r") as file:
        version = file.read()
    return version


def write_version(version: str) -> None:
    with open("src/tensorneko/version.txt", "w") as file:
        file.write(version)
    with open("src/tensorneko_util/version.txt", "w") as file:
        file.write(version)
    with open("src/tensorneko_tool/version.txt", "w") as file:
        file.write(version)


def init_version() -> None:
    version = read_version()
    write_version(version)


if __name__ == '__main__':
    init_version()
