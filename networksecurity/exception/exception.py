import sys
class NetworkSecurityException(Exception): # custom exception class
    def __init__(self,error_message,error_details:sys): # error_details is sys module
        self.error_message = error_message # collect error message
        _,_,exc_tb = error_details.exc_info() # collect exception info
        
        self.lineno=exc_tb.tb_lineno # collect file name
        self.file_name=exc_tb.tb_frame.f_code.co_filename  # collect line number
    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message)) # format the error message
        
if __name__=='__main__':
    try:
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
           raise NetworkSecurityException(e,sys)
