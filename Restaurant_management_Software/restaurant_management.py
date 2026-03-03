"""
Restaurant Management System

This program is designed to help restaurant staff efficiently manage tables, orders, and bills.
It has the following functions:
- Managing table-specific orders and cancellations.
- Applying discounts and special promotions like the Lunch-Special.
- Viewing and saving detailed bills for each table.
- Displaying all currently open bills for review.

The menu is loaded from a CSV file, and the system operates via a console-based interface,
making it easy for staff to use.
"""

__author__ = "7516252, Khaghani"

import csv
import datetime
import os


class MenuItem:
    """
    Represents a menu item in the restaurant.

    :param name: Name of the menu item.
    :type name: str
    :param price: Price of the menu item.
    :type price: float
    """

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __str__(self):
        """
        Returns a string representation of the menu item.

        :return: String in the format "name - price €".
        :type: str
        """
        return f"{self.name} - {self.price:.2f} €"


class OrderItem:
    """
    Represents an individual order item for a table.

    :param menu_item: The menu item being ordered.
    :type menu_item: MenuItem
    :param special_requests: List of special requests for the item.
    :type special_requests: list[str]
    """

    def __init__(self, menu_item, special_requests=None):
        self.menu_item = menu_item
        self.special_requests = special_requests or []
        self.price = menu_item.price + sum(
            1.0 for req in self.special_requests if "extra" in req.lower()
        )

    def __str__(self):
        """
        Returns a string representation of the order item.

        :return: String in the format "name (special_requests) - price €".
        :type: str
        """
        special_requests = ", ".join(self.special_requests) if self.special_requests else "Keine"
        return f"{self.menu_item.name} ({special_requests}) - {self.price:.2f} €"


class Table:
    """
    Represents a table in the restaurant.

    :param table_number: The number identifying the table.
    :type table_number: int
    """

    def __init__(self, table_number):
        self.table_number = table_number
        self.orders = []
        self.discount = 0.0
        self.lunch_special = False

    def add_order(self, order_item):
        """
        Adds an order item to the table's list of orders.

        :param order_item: The order item to add.
        :type order_item: OrderItem
        """
        self.orders.append(order_item)

    def cancel_order(self, index):
        """
        Removes an order item from the table's list of orders.
        """
        if 0 <= index < len(self.orders):
            removed_item = self.orders.pop(index)
            print(f"{removed_item.menu_item.name} wurde storniert.")
        else:
            print("Ungültige Nummer. Keine Bestellung wurde storniert.")

    def apply_discount(self, discount):
        """
        Applies a percentage discount to the table's total.

        :param discount: The discount percentage to apply.
        :type discount: float
        """
        self.discount = discount

    def enable_lunch_special(self):
        """
        Enables a 10% Lunch-Special discount for the table.
        """
        self.lunch_special = True
        print("Lunch-Special aktiviert: 10% Rabatt auf alle Bestellungen.")

    def disable_lunch_special(self):
        """
        Disables the Lunch-Special discount for the table.
        """
        self.lunch_special = False
        print("Lunch-Special deaktiviert.")

    def calculate_total(self):
        """
        Calculates the total cost of all orders, including any discounts.

        :return: A tuple containing the total cost and savings.
        :type: tuple(float, float)
        """
        total = sum(order.price for order in self.orders)
        savings = 0.0
        if self.lunch_special:
            savings += total * 0.1  # 10% Lunch-Special Rabatt
            total *= 0.9
        total *= (1 - self.discount / 100)
        return round(total, 2), round(savings, 2)

    def __str__(self):
        """
        Returns a string representation of the table's orders and total.

        :return: String summarizing the table's orders and total cost.
        :type: str
        """
        total, savings = self.calculate_total()
        order_details = "\n".join(
            [f"{i + 1}. {order}" for i, order in enumerate(self.orders)]
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = (
            f"Tisch {self.table_number} - {timestamp}:\n"
            f"{order_details}\n"
            f"Gesamt: {total:.2f} €"
        )
        if self.lunch_special:
            result += (
                f"\nDurch das Lunch-Special wurden {savings:.2f} € von der Rechnung abgezogen."
            )
        return result


class Restaurant:
    """
    Manages the overall restaurant operations, including tables and menu.

    :param menu_file: Path to the CSV file containing the menu.
    :type menu_file: str
    """

    def __init__(self, menu_file):
        self.tables = {}
        self.menu = self.load_menu(menu_file)

    def load_menu(self, menu_file):
        """
        Loads the menu from a CSV file.

        :param menu_file: Path to the CSV file.
        :type menu_file: str
        :return: A list of MenuItem objects.
        :type: list[MenuItem]
        """
        menu = []
        with open(menu_file, "r") as file:
            reader = csv.reader(file, delimiter=';')  # https://docs.python.org/3/library/csv.html
            next(reader)  # Skip header row
            for row in reader:
                name, _, _, price = row
                price = float(price.replace(',', '.'))
                menu.append(MenuItem(name, price))
        return menu

    def get_table(self, table_number):
        """
        Retrieves a table by its number, creating it if necessary.

        :param table_number: The table number to retrieve.
        :type table_number: int
        :return: The Table object.
        :type: Table
        """
        if table_number not in self.tables:
            self.tables[table_number] = Table(table_number)
        return self.tables[table_number]

    def generate_unique_filename(self, table_number):
        """
        Generates a unique filename for saving a table's bill.

        :param table_number: The table number.
        :type table_number: int
        :return: The unique filename.
        :type: str
        """
        base_filename = f"Tisch_{table_number}_Rechnung_"
        index = 1
        while os.path.exists(f"{base_filename}{index}.txt"):
            index += 1
        return f"{base_filename}{index}.txt"

    def save_and_pay_bill(self, table_number):
        """
        Saves and finalizes the bill for a table, marking it as paid.

        :param table_number: The table number whose bill is being finalized.
        :type table_number: int
        """
        table = self.get_table(table_number)
        total, _ = table.calculate_total()
        filename = self.generate_unique_filename(table_number)
        with open(filename, "w") as file:
            file.write(str(table))
        print(f"Rechnung für Tisch {table_number} bezahlt und gespeichert als {filename}")
        print(f"Betrag: {total:.2f} € wurde bezahlt.")
        del self.tables[table_number]  # Tisch leeren

    def display_open_bills(self):
        """
        Displays all open bills for all tables.
        """
        print("\nOffene Rechnungen:")
        for table_number, table in self.tables.items():
            total, _ = table.calculate_total()
            order_details = "\n  ".join([str(order) for order in table.orders])
            print(f"Tisch {table_number}:\n  {order_details}\n  Gesamt: {total:.2f} €")

    def display_menu(self):
        """
        Displays the restaurant's menu.
        """
        print("Speisekarte:")
        for i, item in enumerate(self.menu):
            print(f"{i + 1}. {item}")

    def place_order(self, table_number):
        """
        Allows the user to place an order for a specific table.

        :param table_number: The number of the table where the order is being placed.
        :type table_number: int
        """
        table = self.get_table(table_number)
        while True:
            self.display_menu()
            choice = self.get_menu_choice()
            if choice.lower() == 'q':
                break
            order_item = self.create_order_item(choice)
            if order_item:
                table.add_order(order_item)
                print(f"{order_item.menu_item.name} wurde hinzugefügt.")
            else:
                print("Ungültige Auswahl.")

    def get_menu_choice(self):
        """
        Prompts the user to choose a menu item or exit.

        :return: The user's menu choice or 'q' to quit.
        :type: str
        """
        return input("Wählen Sie eine Menü-Nummer oder 'q' zum Beenden: ")

    def create_order_item(self, choice):
        """
        Creates an OrderItem object based on user input.

        :param choice: The menu choice selected by the user.
        :type choice: str
        :return: An OrderItem object if the choice is valid, otherwise None.
        :type: OrderItem or None
        """
        try:
            menu_index = int(choice) - 1
            if 0 <= menu_index < len(self.menu):
                special_requests = input(
                    "Sonderwünsche (durch Komma getrennt, oder leer lassen): ").split(",")
                special_requests = [req.strip() for req in special_requests if req.strip()]
                return OrderItem(self.menu[menu_index], special_requests)
        except ValueError:
            pass
        return None

    def manage_table(self, table_number):
        """
        Manages a table, allowing the user to add orders, apply discounts, and more.

        :param table_number: The table number to manage.
        :type table_number: int
        """
        table = self.get_table(table_number)
        while True:
            print(table)
            print("1. Bestellung hinzufügen")
            print("2. Bestellung stornieren")
            print("3. Rabatt hinzufügen")
            print("4. Lunch-Special aktivieren")
            print("5. Lunch-Special deaktivieren")
            print("6. Rechnung bezahlen und speichern")
            print("7. Zurück")
            choice = input("Wählen Sie eine Option: ")
            if choice == "1":
                self.place_order(table_number)
            elif choice == "2":
                try:
                    cancel_index = int(input("Bestellungsnummer zum Stornieren: ")) - 1
                    table.cancel_order(cancel_index)
                except ValueError:
                    print("Ungültige Eingabe.")
            elif choice == "3":
                try:
                    discount = float(input("Rabatt in Prozent: "))
                    table.apply_discount(discount)
                    print(f"Rabatt von {discount}% hinzugefügt.")
                except ValueError:
                    print("Ungültige Eingabe.")
            elif choice == "4":
                table.enable_lunch_special()
            elif choice == "5":
                table.disable_lunch_special()
            elif choice == "6":
                self.save_and_pay_bill(table_number)
                break
            elif choice == "7":
                break
            else:
                print("Ungültige Auswahl.")

    def run(self):
        """
        Runs the main console interface for the restaurant management system.
        """
        print("\nWillkommen im Restaurant-Management-System!")
        print("===========================================")
        print("Hier können Sie Bestellungen aufnehmen, Rechnungen erstellen und verwalten.\n")

        while True:
            print("Hauptmenü:")
            print("1. Tisch verwalten")
            print("2. Offene Rechnungen anzeigen")
            print("3. Programm beenden")

            choice = input("Bitte wählen Sie eine Option (1-3): ")

            if choice == "1":
                while True:
                    try:
                        table_number = int(input("Tischnummer (1-20): "))
                        if 1 <= table_number <= 20:
                            self.manage_table(table_number)
                            break
                        else:
                            print("Ungültige Tischnummer. "
                                  "Bitte geben Sie eine Nummer zwischen 1 und 20 ein.")

                    except ValueError:
                        print("Ungültige Eingabe. Bitte geben Sie eine Zahl ein.")

            elif choice == "2":
                self.display_open_bills()

            elif choice == "3":
                print("\nProgramm beendet. Vielen Dank für die Nutzung des Systems!")
                print("===========================================\n")
                break

            else:
                print("Ungültige Eingabe. Bitte wählen Sie eine Option von 1 bis 3.")


if __name__ == "__main__":
    restaurant = Restaurant("food.csv")
    restaurant.run()
