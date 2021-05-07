"""Example of using an OOP property"""
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

    # Add a decorated balance() method returning _balance        
    @property
    def balance(self):
        return self._balance

    # Add a setter balance() method
    @balance.setter
    def balance(self, new_bal):
        # Validate the parameter value
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal
        print("Setter method called")

# Create a Customer        
cust = Customer('Belinda Lutz', 2000)

# Assign 3000 to the balance property
cust.balance = 3000

# Print the balance property
print(cust.balance)


import pandas as pd
from datetime import datetime

"""Read only property"""
# MODIFY the class to turn created_at into a read-only property
class LoggedDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self._created_at = datetime.today()

    @property
    def created_at(self):
        return(self._created_at)

    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self.created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)   

ldf = LoggedDF({"col1": [1,2], "col2":[3,4]}) 

# Put into try-except block to catch AtributeError and print a message
try:
    ldf.created_at = '2035-07-13'
except AttributeError:
    print('Could not set attribute!')
    