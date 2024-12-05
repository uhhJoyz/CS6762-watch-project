import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
/**
 *
 * @author mm5gg
 */
public class MyWekaUtils {
  public static double classify(String arffData, int option) throws Exception {
    StringReader strReader = new StringReader(arffData);
    Instances instances = new Instances(strReader);
    strReader.close();
    instances.setClassIndex(instances.numAttributes() - 1);
    Classifier classifier;
    if (option == 1)
      classifier = new J48(); // Decision Tree classifier
    else if (option == 2)
      classifier = new RandomForest();
    else if (option == 3)
      classifier = new SMO(); // This is a SVM classifier
    else
      return -1;
    classifier.buildClassifier(instances); // build classifier
    Evaluation eval = new Evaluation(instances);
    eval.crossValidateModel(classifier, instances, 10, new Random(1),
                            new Object[] {});
    return eval.pctCorrect();
  }
  public static String[][] readCSV(String filePath) throws Exception {
    StringBuilder sb = new StringBuilder();
    BufferedReader br = new BufferedReader(new FileReader(filePath));
    ArrayList<String> lines = new ArrayList<String>();
    String line;
    while ((line = br.readLine()) != null) {
      lines.add(line);
      ;
    }
    if (lines.size() == 0) {
      System.out.println("No data found");
      return null;
    }
    int lineCount = lines.size();
    String[][] csvData = new String[lineCount][];
    String[] vals;
    int i, j;
    for (i = 0; i < lineCount; i++) {
      csvData[i] = lines.get(i).split(",");
    }
    return csvData;
  }
  public static String csvToArff(String[][] csvData, int[] featureIndices)
      throws Exception {
    int total_rows = csvData.length;
    int total_cols = csvData[0].length;
    int fCount = featureIndices.length;
    String[] attributeList = new String[fCount + 1];
    int i, j;
    for (i = 0; i < fCount; i++) {
      attributeList[i] = csvData[0][featureIndices[i]];
    }
    attributeList[i] = csvData[0][total_cols - 1];
    String[] classList = new String[1];
    classList[0] = csvData[1][total_cols - 1];
    for (i = 1; i < total_rows; i++) {
      classList = addClass(classList, csvData[i][total_cols - 1]);
    }
    StringBuilder sb = getArffHeader(attributeList, classList);
    for (i = 1; i < total_rows; i++) {
      for (j = 0; j < fCount; j++) {
        sb.append(csvData[i][featureIndices[j]]);
        sb.append(",");
      }
      sb.append(csvData[i][total_cols - 1]);
      sb.append("\n");
    }
    return sb.toString();
  }
  private static StringBuilder getArffHeader(String[] attributeList,
                                             String[] classList) {
    StringBuilder s = new StringBuilder();
    s.append("@RELATION wada\n\n");
    int i;
    for (i = 0; i < attributeList.length - 1; i++) {
      s.append("@ATTRIBUTE ");
      s.append(attributeList[i]);
      s.append(" numeric\n");
    }
    s.append("@ATTRIBUTE ");
    s.append(attributeList[i]);
    s.append(" {");
    s.append(classList[0]);
    for (i = 1; i < classList.length; i++) {
      s.append(",");
      s.append(classList[i]);
    }
    s.append("}\n\n");
    s.append("@DATA\n");
    return s;
  }
  private static String[] addClass(String[] classList, String className) {
    int len = classList.length;
    int i;
    for (i = 0; i < len; i++) {
      if (className.equals(classList[i])) {
        return classList;
      }
    }
    String[] newList = new String[len + 1];
    for (i = 0; i < len; i++) {
      newList[i] = classList[i];
    }
    newList[i] = className;
    return newList;
  }

  private static int[] featureSelection(String[][] features, int[] featureInd,
                                        int option) {
    double best_acc = 0;
    double prev_acc = 0;
    ArrayList<Integer> feature_list = new ArrayList<Integer>(Arrays.asList(
        Arrays.stream(featureInd).boxed().toArray(Integer[] ::new)));
    ArrayList<Integer> sel_features = new ArrayList<Integer>();
    int m = -1;
    do {
      m = -1;
      double max_acc = best_acc;
      for (int i = 0; i < feature_list.size(); i++) {
        try {
          sel_features.add(feature_list.get(i));
          String arff = csvToArff(
              features, sel_features.stream().mapToInt(j -> j).toArray());
          double acc = classify(arff, option);
          if (acc > max_acc) {
            m = feature_list.get(i);
            max_acc = acc;
          }
          sel_features.remove(sel_features.size() - 1);
        } catch (Exception e) {
          System.out.println(e);
        }
      }
      // at the end of the loop, set up for the next iteration of the while loop
      if (m != -1) {
        feature_list.remove(feature_list.indexOf(m));
        sel_features.add(m);
        prev_acc = best_acc;
        best_acc = max_acc;
        System.out.println(best_acc);
      }
    } while (best_acc - prev_acc > 0.01 && m != -1 &&
             sel_features.size() < featureInd.length);

    return sel_features.stream().mapToInt(i -> i).toArray();
  }

  public static void main(String[] args) {
    // Disable warnings
    System.err.close();
    System.setErr(System.out);

    String[] files = new String[] {"../processed_door_data/features_1.csv",
                                   "../processed_door_data/features_2.csv",
                                   "../processed_door_data/features_3.csv"};
    for (int i = 0; i < 3; i++) {
      for (int j = 1; j < 4; j++) {
        String[][] features;
        String arff;
        try {
          System.out.print("Staring work on ");
          System.out.println(files[i]);
          System.out.print("Classifier option: ");
          System.out.println(String.valueOf(j));
          features = readCSV(files[i]);
          int[] sel_features = featureSelection(
              features, IntStream.rangeClosed(0, 11).toArray(), j);
          arff = csvToArff(features, sel_features);
          System.out.print("Features selected: ");
          for (int k = 0; k < sel_features.length; k++) {
            System.out.print(sel_features[k]);
            System.out.print(" ");
          }
          System.out.println();
          double acc = classify(arff, j);
          System.out.print("The classifier accuracy is: ");
          System.out.println(acc);
        } catch (Exception e) {
          System.out.println(e);
        }
      }
    }
  }
}
