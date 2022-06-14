/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Table of Contents
*/
/*:
 # Creating a Model from Tabular Data
 ## Table of Contents
 - [Train a Regressor and a Classifier](Train%20a%20Regressor%20and%20a%20Classifier)
 - [LICENSE](LICENSE)
 ****
 [Next](@next)
 */
import CreateML
import Foundation

let csvFile = Bundle.main.url(forResource: "twitter-sanders-apple3", withExtension: "csv")!
let dataTable = try MLDataTable(contentsOf: csvFile)

let (trainingData, testingData) = dataTable.randomSplit(by: 0.8, seed: 5)
