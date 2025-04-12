from typing import Dict, List, Tuple
import copy

#We define a calculator class to manage group transactions.
class GroupTransactionCalculator:
    def __init__(self):
        self.personPayments: Dict[str, float] = {}  #We track each person's total payment amount.
        self.itemRecords: List[Tuple[str, float, Dict[str, int], bool]] = []  #We record each item as (buyer, cost, {person: quantity}, isCommon).
        self.settlementTransactions: List[str] = []  #We store the suggested settlement transactions.
        self.detailedDebts: Dict[Tuple[str, str], float] = {}  #We record detailed debts mapping (debtor, creditor) to owed amount.

    #We add a person if they're not already in my records.
    def addPerson(self, personName: str) -> None:
        if personName not in self.personPayments:  #We ensure the person is tracked.
            self.personPayments[personName] = 0.0  #We initialize their total payment to 0.

    #We add an item with its cost and involved people, tracking the debts accordingly.
    def addItem(self, buyer: str, cost: float, peopleInvolved: Dict[str, int], isCommon: bool = False) -> None:
        self.addPerson(buyer)  #We ensure the buyer is in my records.
        self.personPayments[buyer] += cost  #We update the buyer's total payment.
        for person in peopleInvolved:  #We make sure every involved person is added.
            self.addPerson(person)
        self.itemRecords.append((buyer, cost, peopleInvolved, isCommon))  #We record the item details.

        #We now track detailed debts based on whether the item is common or individual.
        if isCommon:
            share = cost / len(peopleInvolved)  #We calculate an equal share.
            for person in peopleInvolved:  #We process each involved person.
                if person != buyer:  #We skip the buyer.
                    self.detailedDebts[(person, buyer)] = self.detailedDebts.get((person, buyer), 0) + share  #We add the share owed.
        else:
            totalQuantity = sum(peopleInvolved.values())  #We compute the total quantity for proportional splitting.
            if totalQuantity == 0:  #We check for a possible division by zero.
                return  #We exit if there is nothing to split.
            costPerUnit = cost / totalQuantity  #We calculate cost per unit.
            for person, quantity in peopleInvolved.items():  #We go through each person and quantity.
                amountOwed = costPerUnit * quantity  #We compute the owed amount for the person.
                if person != buyer:  #We skip the buyer.
                    self.detailedDebts[(person, buyer)] = self.detailedDebts.get((person, buyer), 0) + amountOwed  #We record the owed amount.

    #We calculate how much each person should pay based on all items.
    def calculateShares(self) -> Dict[str, float]:
        owedAmounts: Dict[str, float] = {person: 0.0 for person in self.personPayments}  #We initialize owed amounts for everyone.
        for buyer, cost, peopleInvolved, isCommon in self.itemRecords:  #We iterate over all recorded items.
            if isCommon:
                share = cost / len(peopleInvolved)  #We compute equal share for common items.
                for person in peopleInvolved:  #We add the share to each person.
                    owedAmounts[person] += share
            else:
                totalQuantity = sum(peopleInvolved.values())  #We determine total quantity for individual items.
                if totalQuantity == 0:  #We skip if no quantity is provided.
                    continue
                costPerUnit = cost / totalQuantity  #We calculate the cost per unit.
                for person, quantity in peopleInvolved.items():  #We add the proportional cost per person.
                    owedAmounts[person] += costPerUnit * quantity
        return owedAmounts  #We return the computed owed amounts.

    #We optimize the transactions needed to settle debts using minimax with alpha-beta pruning.
    def optimizeTransactions(self, owedAmounts: Dict[str, float]) -> None:
        balances = {person: self.personPayments[person] - owedAmounts.get(person, 0) for person in self.personPayments}  #We compute net balances.

        #We round very small balance values to 0.
        for person in balances:
            if abs(balances[person]) < 0.01:  #We check for negligible balances.
                balances[person] = 0.0  #We set them to 0.

        debtorsList = [(person, balance) for person, balance in balances.items() if balance < 0]  #We list those who owe money.
        creditorsList = [(person, balance) for person, balance in balances.items() if balance > 0]  #We list those owed money.

        if not debtorsList or not creditorsList:  #We check if there are any unsettled debts.
            self.settlementTransactions = []  #We clear transactions if all are settled.
            return  #We exit if nothing needs to be done.

        #We define a recursive function to explore settlement transactions using minimax with pruning.
        def evaluateState(debtors: List[Tuple[str, float]],
                          creditors: List[Tuple[str, float]],
                          transactions: List[str],
                          depth: int,
                          alpha: int,
                          beta: int,
                          maximizing: bool) -> Tuple[int, List[str]]:
            if not debtors or not creditors:  #We check if all debts are settled.
                return len(transactions), transactions  #We return the transaction count and list.

            if maximizing:  #We try to minimize the number of transactions.
                bestScore = float('inf')  #We initialize the best score as infinity.
                bestTransactions = transactions.copy()  #We copy the current transaction list.
                for i, (debtorName, debtorAmount) in enumerate(debtors):  #We iterate over each debtor.
                    for j, (creditorName, creditorAmount) in enumerate(creditors):  #We iterate over each creditor.
                        if debtorName == creditorName:  #We skip if they are the same person.
                            continue
                        amount = min(-debtorAmount, creditorAmount)  #We calculate the maximum transferable amount.
                        newTransaction = f"{debtorName} pays {creditorName} ${amount:.2f}"  #We format the transaction.
                        newDebtors = debtors.copy()  #We copy the current list of debtors.
                        newCreditors = creditors.copy()  #We copy the current list of creditors.
                        newTransactions = transactions + [newTransaction]  #We add the new transaction to the list.
                        newDebtors[i] = (debtorName, debtorAmount + amount)  #We update the debtor's balance.
                        newCreditors[j] = (creditorName, creditorAmount - amount)  #We update the creditor's balance.
                        if abs(newDebtors[i][1]) < 0.01:  #We check if the debtor's balance is effectively zero.
                            newDebtors.pop(i)  #We remove the settled debtor.
                        if abs(newCreditors[j][1]) < 0.01:  #We check if the creditor's balance is settled.
                            newCreditors.pop(j)  #We remove the settled creditor.
                        score, trans = evaluateState(newDebtors, newCreditors, newTransactions, depth + 1, alpha, beta, False)  #We recurse into the next state.
                        if score < bestScore:  #We update the best score if a better solution is found.
                            bestScore = score  #We record the new best score.
                            bestTransactions = trans  #We record the corresponding transaction list.
                        beta = min(beta, bestScore)  #We update beta for pruning.
                        if beta <= alpha:  #We check if I can prune this branch.
                            break  #We break out if pruning is applicable.
                    if beta <= alpha:  #We prune at the outer loop level if needed.
                        break  #We exit the loop.
                return bestScore, bestTransactions  #We return the best solution found.
            else:
                #We simulate the opponent's move by flipping the maximizing flag.
                return evaluateState(debtors, creditors, transactions, depth, alpha, beta, True)

        #We invoke the recursive function to set my settlementTransactions.
        _, self.settlementTransactions = evaluateState(debtorsList, creditorsList, [], 0, float('-inf'), float('inf'), True)

    #We display the results including contributions, item details, detailed debts, owed amounts, balances, and optimized transactions.
    def displayResults(self) -> None:
        print("\nContributions (Amount Paid):")
        for person, amount in self.personPayments.items():  #We print each person's total contribution.
            print(f"{person}: ${amount:.2f}")

        print("\nItem Details:")
        for idx, (buyer, cost, peopleInvolved, isCommon) in enumerate(self.itemRecords, 1):  #We enumerate item records.
            if isCommon:
                involvedStr = " ".join(peopleInvolved.keys())  #We join the names for common items.
                print(f"Item {idx}: {buyer} paid ${cost:.2f}, Common, Split among: {involvedStr}")  #We print common item details.
            else:
                totalQty = sum(peopleInvolved.values())  #We calculate the total quantity for individual items.
                qtyStr = " ".join(f"{p}: {q}" for p, q in peopleInvolved.items())  #We format the distribution details.
                print(f"Item {idx}: {buyer} paid ${cost:.2f}, Individual, Total Qty: {totalQty}, Distribution: {qtyStr}")  #We print individual item details.

        print("\nDetailed Debts (Who Owes Whom):")
        for (debtor, creditor), amount in self.detailedDebts.items():  #We iterate over detailed debts.
            print(f"{debtor} owes {creditor} ${amount:.2f}")  #We print who owes whom and how much.

        owedAmounts = self.calculateShares()  #We calculate what each person should owe.
        print("\nTotal Amount Owed by Each:")
        for person, amount in owedAmounts.items():  #We print the owed amount for each person.
            print(f"{person}: ${amount:.2f}")

        balances = {person: self.personPayments[person] - owedAmounts.get(person, 0) for person in self.personPayments}  #We compute net balances.
        print("\nBalances (Positive = Owed, Negative = Owes):")
        for person, amount in balances.items():  #We print each person's net balance.
            if abs(amount) < 0.01:  #We check if the balance is negligible.
                print(f"{person}: $0.00 (Settled)")  #We show settled balance.
            else:
                print(f"{person}: ${amount:.2f}")  #We show the outstanding balance.

        print("\nOptimized Transactions to Settle (Minimized via Minimax):")
        if not self.settlementTransactions:  #We check if any transactions are needed.
            print("No transactions needed; all balances are settled.")  #We indicate if everything is settled.
        for transaction in self.settlementTransactions:  #We print each optimized transaction.
            print(transaction)

#We define the main function to run my group transaction calculator.
def main():
    calc = GroupTransactionCalculator()  #We instantiate the calculator.

    print("Add items (format: 'buyer cost person1 qty1 person2 qty2 ...' for individual, 'buyer cost person1 person2 ... common' for common, 'done' when finished):")  #We instruct the user on input format.
    print("Examples: 'Alice 30 Alice 1 Bob 2 Charlie 1' or 'Bob 15 Alice Bob Charlie common'")  #We provide example inputs.
    while True:
        entry = input("> ").strip()  #We get user input.
        if entry.lower() == "done":  #We allow the user to exit.
            break  #We exit the loop.
        try:
            parts = entry.split()  #We split the input into parts.
            buyer = parts[0]  #We identify the buyer.
            cost = float(parts[1])  #We convert the cost to a float.
            peopleInvolved = {}  #We initialize a dictionary for people involved.
            isCommon = "common" in parts[-1].lower()  #We determine if the item is common.
            if isCommon:
                for person in parts[2:-1]:  #We iterate over names for common items.
                    peopleInvolved[person] = 1  #We assign a default quantity of 1.
            else:
                i = 2  #We start processing from the third element.
                while i < len(parts):
                    person = parts[i]  #We get the person's name.
                    qty = int(parts[i + 1])  #We get the corresponding quantity.
                    peopleInvolved[person] = qty  #We store the quantity.
                    i += 2  #We move to the next pair.
            calc.addItem(buyer, cost, peopleInvolved, isCommon)  #We add the item to my calculator.
        except (ValueError, IndexError):  #We handle any input errors.
            print("Invalid input. Format: 'buyer cost person1 qty1 ...' or 'buyer cost person1 person2 ... common'")  #We inform the user of the correct format.

    owedAmounts = calc.calculateShares()  #We calculate how much each person should owe.
    calc.optimizeTransactions(owedAmounts)  #We optimize the settlement transactions.
    calc.displayResults()  #We display all the results.

if __name__ == "__main__":
    main()  #We run the main function.
