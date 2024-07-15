import argparse

from .utils import version

def main():
    parser = argparse.ArgumentParser(description="TensorNeko CLI Tools")
    parser.add_argument("--version", action="store_true")

    sub_parser = parser.add_subparsers(dest="sub_command")
    parser_gotify = sub_parser.add_parser("gotify", help="Send message to Gotify server")
    parser_gotify.add_argument("message", help="The message to send", type=str)
    parser_gotify.add_argument("--title", help="The title of the message", type=str)
    parser_gotify.add_argument("--priority", help="The priority of the message", type=int, default=0)
    parser_gotify.add_argument("--url", help="The URL of the Gotify server", type=str)
    parser_gotify.add_argument("--token", help="The token of the Gotify server", type=str)

    args = parser.parse_args()

    if args.version:
        print(version.value)
        exit(0)

    if args.sub_command == "gotify":
        from tensorneko_util.msg.gotify import push_gotify
        push_gotify(args.message, args.url, args.token, args.title, args.priority)
        exit(0)


if __name__ == "__main__":
    main()
