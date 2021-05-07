"""Python allows you to define class methods as well, using the @classmethod decorator and a special first argument cls. The main use of class methods is defining methods that return an instance of the class, but aren't using the same code as __init__().

For example, you are developing a time series package and want to define your own class for working with dates, BetterDate. The attributes of the class will be year, month, and day. You want to have a constructor that creates BetterDate objects given the values for year, month, and day, but you also want to be able to create BetterDate objects from strings like 2020-04-30.

You might find the following functions useful:

.split("-") method will split a string at"-" into an array, e.g. "2020-04-30".split("-") returns ["2020", "04", "30"],
int() will convert a string into a number, e.g. int("2019") is 2019 ."""


class BetterDate:    
    # Constructor
    def __init__(self, year, month, day):
      # Recall that Python allows multiple variable assignments in one line
      self.year, self.month, self.day = year, month, day
    
    # Define a class method from_str
    @classmethod
    def from_str(cls, datestr: str):
        # Split the string at "-" and convert each part to integer
        parts = datestr.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        # Return the class instance
        return BetterDate(year, month, day)
        
bd = BetterDate.from_str('2020-04-30')   
print(bd.year)
print(bd.month)
print(bd.day)