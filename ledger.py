from typing import Dict, List, Tuple
import copy

# I define a calculator class to manage group transactions.
class GroupTransactionCalculator:
    def __init__(self):
        self.personPayments: Dict[str, float] = {}  # I track each person's total payment amount.
        self.itemRecords: List[Tuple[str, float, Dict[str, int], bool]] = []  # I record each item as (buyer, cost, {person: quantity}, isCommon).
        self.settlementTransactions: List[str] = []  # I store the suggested settlement transactions.
        self.detailedDebts: Dict[Tuple[str, str], float] = {}  # I record detailed debts mapping (debtor, creditor) to owed amount.

    # I add a person if they're not already in my records.
    def addPerson(self, personName: str) -> None:
        if personName not in self.personPayments:  # I ensure the person is tracked.
            self.personPayments[personName] = 0.0  # I initialize their total payment to 0.

    # I add an item with its cost and involved people, tracking the debts accordingly.
    def addItem(self, buyer: str, cost: float, peopleInvolved: Dict[str, int], isCommon: bool = False) -> None:
        self.addPerson(buyer)  # I ensure the buyer is in my records.
        self.personPayments[buyer] += cost  # I update the buyer's total payment.
        for person in peopleInvolved:  # I make sure every involved person is added.
            self.addPerson(person)
        self.itemRecords.append((buyer, cost, peopleInvolved, isCommon))  # I record the item details.

        # I now track detailed debts based on whether the item is common or individual.
        if isCommon:
            share = cost / len(peopleInvolved)  # I calculate an equal share.
            for person in peopleInvolved:  # I process each involved person.
                if person != buyer:  # I skip the buyer.
                    self.detailedDebts[(person, buyer)] = self.detailedDebts.get((person, buyer), 0) + share  # I add the share owed.
        else:
            totalQuantity = sum(peopleInvolved.values())  # I compute the total quantity for proportional splitting.
            if totalQuantity == 0:  # I check for a possible division by zero.
                return  # I exit if there is nothing to split.
            costPerUnit = cost / totalQuantity  # I calculate cost per unit.
            for person, quantity in peopleInvolved.items():  # I go through each person and quantity.
                amountOwed = costPerUnit * quantity  # I compute the owed amount for the person.
                if person != buyer:  # I skip the buyer.
                    self.detailedDebts[(person, buyer)] = self.detailedDebts.get((person, buyer), 0) + amountOwed  # I record the owed amount.

    # I calculate how much each person should pay based on all items.
    def calculateShares(self) -> Dict[str, float]:
        owedAmounts: Dict[str, float] = {person: 0.0 for person in self.personPayments}  # I initialize owed amounts for everyone.
        for buyer, cost, peopleInvolved, isCommon in self.itemRecords:  # I iterate over all recorded items.
            if isCommon:
                share = cost / len(peopleInvolved)  # I compute equal share for common items.
                for person in peopleInvolved:  # I add the share to each person.
                    owedAmounts[person] += share
            else:
                totalQuantity = sum(peopleInvolved.values())  # I determine total quantity for individual items.
                if totalQuantity == 0:  # I skip if no quantity is provided.
                    continue
                costPerUnit = cost / totalQuantity  # I calculate the cost per unit.
                for person, quantity in peopleInvolved.items():  # I add the proportional cost per person.
                    owedAmounts[person] += costPerUnit * quantity
        return owedAmounts  # I return the computed owed amounts.

    # I optimize the transactions needed to settle debts using minimax with alpha-beta pruning.
    def optimizeTransactions(self, owedAmounts: Dict[str, float]) -> None:
        balances = {person: self.personPayments[person] - owedAmounts.get(person, 0) for person in self.personPayments}  # I compute net balances.

        # I round very small balance values to 0.
        for person in balances:
            if abs(balances[person]) < 0.01:  # I check for negligible balances.
                balances[person] = 0.0  # I set them to 0.

        debtorsList = [(person, balance) for person, balance in balances.items() if balance < 0]  # I list those who owe money.
        creditorsList = [(person, balance) for person, balance in balances.items() if balance > 0]  # I list those owed money.

        if not debtorsList or not creditorsList:  # I check if there are any unsettled debts.
            self.settlementTransactions = []  # I clear transactions if all are settled.
            return  # I exit if nothing needs to be done.

        # I define a recursive function to explore settlement transactions using minimax with pruning.
        def evaluateState(debtors: List[Tuple[str, float]],
                          creditors: List[Tuple[str, float]],
                          transactions: List[str],
                          depth: int,
                          alpha: int,
                          beta: int,
                          maximizing: bool) -> Tuple[int, List[str]]:
            if not debtors or not creditors:  # I check if all debts are settled.
                return len(transactions), transactions  # I return the transaction count and list.

            if maximizing:  # I try to minimize the number of transactions.
                bestScore = float('inf')  # I initialize the best score as infinity.
                bestTransactions = transactions.copy()  # I copy the current transaction list.
                for i, (debtorName, debtorAmount) in enumerate(debtors):  # I iterate over each debtor.
                    for j, (creditorName, creditorAmount) in enumerate(creditors):  # I iterate over each creditor.
                        if debtorName == creditorName:  # I skip if they are the same person.
                            continue
                        amount = min(-debtorAmount, creditorAmount)  # I calculate the maximum transferable amount.
                        newTransaction = f"{debtorName} pays {creditorName} ${amount:.2f}"  # I format the transaction.
                        newDebtors = debtors.copy()  # I copy the current list of debtors.
                        newCreditors = creditors.copy()  # I copy the current list of creditors.
                        newTransactions = transactions + [newTransaction]  # I add the new transaction to the list.
                        newDebtors[i] = (debtorName, debtorAmount + amount)  # I update the debtor's balance.
                        newCreditors[j] = (creditorName, creditorAmount - amount)  # I update the creditor's balance.
                        if abs(newDebtors[i][1]) < 0.01:  # I check if the debtor's balance is effectively zero.
                            newDebtors.pop(i)  # I remove the settled debtor.
                        if abs(newCreditors[j][1]) < 0.01:  # I check if the creditor's balance is settled.
                            newCreditors.pop(j)  # I remove the settled creditor.
                        score, trans = evaluateState(newDebtors, newCreditors, newTransactions, depth + 1, alpha, beta, False)  # I recurse into the next state.
                        if score < bestScore:  # I update the best score if a better solution is found.
                            bestScore = score  # I record the new best score.
                            bestTransactions = trans  # I record the corresponding transaction list.
                        beta = min(beta, bestScore)  # I update beta for pruning.
                        if beta <= alpha:  # I check if I can prune this branch.
                            break  # I break out if pruning is applicable.
                    if beta <= alpha:  # I prune at the outer loop level if needed.
                        break  # I exit the loop.
                return bestScore, bestTransactions  # I return the best solution found.
            else:
                # I simulate the opponent's move by flipping the maximizing flag.
                return evaluateState(debtors, creditors, transactions, depth, alpha, beta, True)

        # I invoke the recursive function to set my settlementTransactions.
        _, self.settlementTransactions = evaluateState(debtorsList, creditorsList, [], 0, float('-inf'), float('inf'), True)

    # I display the results including contributions, item details, detailed debts, owed amounts, balances, and optimized transactions.
    def displayResults(self) -> None:
        print("\nContributions (Amount Paid):")
        for person, amount in self.personPayments.items():  # I print each person's total contribution.
            print(f"{person}: ${amount:.2f}")

        print("\nItem Details:")
        for idx, (buyer, cost, peopleInvolved, isCommon) in enumerate(self.itemRecords, 1):  # I enumerate item records.
            if isCommon:
                involvedStr = " ".join(peopleInvolved.keys())  # I join the names for common items.
                print(f"Item {idx}: {buyer} paid ${cost:.2f}, Common, Split among: {involvedStr}")  # I print common item details.
            else:
                totalQty = sum(peopleInvolved.values())  # I calculate the total quantity for individual items.
                qtyStr = " ".join(f"{p}: {q}" for p, q in peopleInvolved.items())  # I format the distribution details.
                print(f"Item {idx}: {buyer} paid ${cost:.2f}, Individual, Total Qty: {totalQty}, Distribution: {qtyStr}")  # I print individual item details.

        print("\nDetailed Debts (Who Owes Whom):")
        for (debtor, creditor), amount in self.detailedDebts.items():  # I iterate over detailed debts.
            print(f"{debtor} owes {creditor} ${amount:.2f}")  # I print who owes whom and how much.

        owedAmounts = self.calculateShares()  # I calculate what each person should owe.
        print("\nTotal Amount Owed by Each:")
        for person, amount in owedAmounts.items():  # I print the owed amount for each person.
            print(f"{person}: ${amount:.2f}")

        balances = {person: self.personPayments[person] - owedAmounts.get(person, 0) for person in self.personPayments}  # I compute net balances.
        print("\nBalances (Positive = Owed, Negative = Owes):")
        for person, amount in balances.items():  # I print each person's net balance.
            if abs(amount) < 0.01:  # I check if the balance is negligible.
                print(f"{person}: $0.00 (Settled)")  # I show settled balance.
            else:
                print(f"{person}: ${amount:.2f}")  # I show the outstanding balance.

        print("\nOptimized Transactions to Settle (Minimized via Minimax):")
        if not self.settlementTransactions:  # I check if any transactions are needed.
            print("No transactions needed; all balances are settled.")  # I indicate if everything is settled.
        for transaction in self.settlementTransactions:  # I print each optimized transaction.
            print(transaction)

# I define the main function to run my group transaction calculator.
def main():
    calc = GroupTransactionCalculator()  # I instantiate the calculator.

    print("Add items (format: 'buyer cost person1 qty1 person2 qty2 ...' for individual, 'buyer cost person1 person2 ... common' for common, 'done' when finished):")  # I instruct the user on input format.
    print("Examples: 'Alice 30 Alice 1 Bob 2 Charlie 1' or 'Bob 15 Alice Bob Charlie common'")  # I provide example inputs.
    while True:
        entry = input("> ").strip()  # I get user input.
        if entry.lower() == "done":  # I allow the user to exit.
            break  # I exit the loop.
        try:
            parts = entry.split()  # I split the input into parts.
            buyer = parts[0]  # I identify the buyer.
            cost = float(parts[1])  # I convert the cost to a float.
            peopleInvolved = {}  # I initialize a dictionary for people involved.
            isCommon = "common" in parts[-1].lower()  # I determine if the item is common.
            if isCommon:
                for person in parts[2:-1]:  # I iterate over names for common items.
                    peopleInvolved[person] = 1  # I assign a default quantity of 1.
            else:
                i = 2  # I start processing from the third element.
                while i < len(parts):
                    person = parts[i]  # I get the person's name.
                    qty = int(parts[i + 1])  # I get the corresponding quantity.
                    peopleInvolved[person] = qty  # I store the quantity.
                    i += 2  # I move to the next pair.
            calc.addItem(buyer, cost, peopleInvolved, isCommon)  # I add the item to my calculator.
        except (ValueError, IndexError):  # I handle any input errors.
            print("Invalid input. Format: 'buyer cost person1 qty1 ...' or 'buyer cost person1 person2 ... common'")  # I inform the user of the correct format.

    owedAmounts = calc.calculateShares()  # I calculate how much each person should owe.
    calc.optimizeTransactions(owedAmounts)  # I optimize the settlement transactions.
    calc.displayResults()  # I display all the results.

if __name__ == "__main__":
    main()  # I run the main function.
