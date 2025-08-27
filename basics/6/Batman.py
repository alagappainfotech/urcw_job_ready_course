# And yet another class definition that inherits from Superhero and Bat
# superhero.py
from human.superhero.Superhero import Superhero
from bat import Bat

# Define Batman as a child that inherits from both Superhero and Bat
class Batman(Superhero, Bat):

    def __init__(self, *args, **kwargs):
        # Typically to inherit attributes you have to call super:
        # super(Batman, self).__init__(*args, **kwargs)
        # However we are dealing with multiple inheritance here, and super()
        # only works with the next base class in the MRO list.
        # So instead we explicitly call __init__ for all ancestors.
        # The use of *args and **kwargs allows for a clean way to pass
        # arguments, with each parent "peeling a layer of the onion".
        Superhero.__init__(self, "anonymous", movie=True,
                           superpowers=["Wealthy"], *args, **kwargs)
        Bat.__init__(self, *args, can_fly=False, **kwargs)
        # override the value for the name attribute
        self.name = "Sad Affleck"

    def sing(self):
        return "nan nan nan nan nan batman!"


if __name__ == "__main__":
    sup = Batman()

    # The Method Resolution Order
    print(Batman.__mro__)     # => (<class '__main__.Batman'>,
                              # => <class 'superhero.Superhero'>,
                              # => <class 'human.Human'>,
                              # => <class 'bat.Bat'>, <class 'object'>)

    # Calls parent method but uses its own class attribute
    print(sup.get_species())  # => Superhuman

    # Calls overridden method
    print(sup.sing())         # => nan nan nan nan nan batman!

    # Calls method from Human, because inheritance order matters
    sup.say("I agree")        # => Sad Affleck: I agree

    # Call method that exists only in 2nd ancestor
    print(sup.sonar())        # => ))) ... (((

    # Inherited class attribute
    sup.age = 100
    print(sup.age)            # => 100

    # Inherited attribute from 2nd ancestor whose default value was overridden.
    print("Can I fly? " + str(sup.fly))  # => Can I fly? False

