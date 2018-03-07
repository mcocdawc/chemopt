class ConvergenceFinished(Exception):
    def __init__(self, successful):
        self.successful = successful

    def __str__(self):
        get_str = 'Converge finished {} successfully'.format
        return get_str('' if self.successful else 'not')

class ElectronicCalculation(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return message
