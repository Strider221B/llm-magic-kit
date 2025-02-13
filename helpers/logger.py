class Logger:
    '''
    For now just prints logs. Will update to use python logger or any other mechanism later as required.
    '''

    @staticmethod
    def debug(msg, *args, **kwargs):
        print(msg)

    @staticmethod
    def info(msg, *args, **kwargs):
        print(msg)

    @staticmethod
    def error(msg, *args, **kwargs):
        print(msg)

    @staticmethod
    def critical(msg, *args, **kwargs):
        print(msg)

    @staticmethod
    def exception(msg, *args, **kwargs):
        print(msg)
