package decision;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.TreeSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DecisionTree extends SupervisedLearner {

	//ArrayList<ArrayList<Node>> tree;
	ArrayList<Node> layer;
	List<Double> commonElement;
	Node root;
	
	public DecisionTree() {
		layer = new ArrayList<Node>();
		root = new Node();
		commonElement = new ArrayList<Double>();
		//tree = new ArrayList<ArrayList<Node>>();
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		Map<Integer, Set<Double>> outClass = constructOutputs(features);
		cleanData(features);
		createTree(features, labels, root, 0, outClass);
		//printTree(root);
	}
	
	public void cleanData(Matrix features) {
		int size = features.m_data.size();
		for (int i = 0; i < size; i++) {
			double[] row = features.m_data.get(i);
			int length = row.length;
			for(int j = 0; j < length; j++) {
				if(row[j] != Double.MAX_VALUE) {
					continue;
				} else {
					features.set(i, j, features.mostCommonValue(j));
				}
			}
		}
		
		int length = features.m_data.get(0).length;
		for(int j = 0; j < length; j++) {
			commonElement.add(features.mostCommonValue(j));
		}
	}
	
	public void createTree(Matrix features, Matrix labels, Node currNode, int childNum, Map<Integer, Set<Double>> outputClasses) {
		int rows = features.rows();
		int cols = features.cols();
		
		// find the possible classification values
		Set<Double> outputs = new TreeSet<Double>();
		List<double[]> tempLabels = labels.m_data;
		// convert the double array list to a simpler double list
		List<Double> targets = new ArrayList<Double>();
		// go through the labels and add them to the set and list
		for (int i = 0; i < rows; i++) {
			double val = tempLabels.get(i)[0];
			outputs.add(val);
			targets.add(val);
		}
		// calculate the info of the output class
		double info = calculateInfo(outputs, targets);
		//System.out.println(info);
		if(info == 0) {
			currNode.setOut(tempLabels.get(0)[0]);
			return;
		}
			
		// transposes the matrix so the features are organized in columns not rows
		List<List<Double>> columnMatrix = new ArrayList<List<Double>>();
		int maxIndex = -1;
		double maxGain = -1000;
		// iterate over all the columns
		 
		for (int i = 0; i < cols; i++) {
			Set<Double> featureVals = outputClasses.get(i);
			List<Double> featureData = new ArrayList<Double>();
			// iterate over all the rows for that column
			for (int j = 0; j < rows; j++) {
				// create a list of all the values and a set of possible outputs
				double val = features.m_data.get(j)[i];
				featureData.add(val);
			}
			columnMatrix.add(featureData);
			Node attr = new Node(featureVals, featureData);
			// calculate the gain using the info
			double gain = calculateGain(attr, outputs, targets, info);
			// update the max gain and max index
			maxIndex = (gain > maxGain) ? i : maxIndex;
			maxGain = (gain > maxGain) ? gain : maxGain;	
		}
		
		
		Node split = layer.get(maxIndex);
		split.setParent(currNode.getParent());
		split.setIndex(maxIndex);
		split.setData(features, labels);
		if (currNode.getParent()==null) {
			root = split;
		} else {
			split.getParent().getChildren().set(childNum, split);
		}
		
		currNode = split;
		Map<Integer, Set<Double>> outCopy = new HashMap<Integer, Set<Double>>();
		outCopy = updateOutputClasses(maxIndex, outputClasses);
		layer = new ArrayList<Node>();
		
		if(columnMatrix.size() != 0) {
			createChildren(split, features.m_data, labels.m_data, columnMatrix, maxIndex);
		} else {
			double common = findMostCommon(targets,split.getVals());
			//Node child = new Node();
			//child.setParent(split);
			split.setOut(common);
			//split.getChildren().set(childNum, child);
			return;
		}
		
		int i = 0;
		for(Node child : split.getChildren()) {
			if(child.getFeatures().cols()==0 || child.getFeatures().rows()==0) {
				double common = child.getParent().getLabels().mostCommonValue(0);
				split.setOut(common);
				continue;
			}
			createTree(child.getFeatures(), child.getLabels(), child, i, outCopy);
			i++;
		}
	}
	
	public void createChildren(Node split, ArrayList<double[]> data, ArrayList<double[]> targets, List<List<Double>> columnMatrix, int index) {
		Set<Double> featureVals = split.getVals();
		int cols = columnMatrix.size();
		int size = data.size();
		List<Node> children = new ArrayList<Node>();
		// go through all the remaining attributes
		for(Double val: featureVals) {
			Node child = new Node();
			// set the parent of the child to be the feature we split on
			child.setParent(split);
			// calculate the number of rows that we should have in the new matrix
			int rows = Collections.frequency(columnMatrix.get(index), val);
			Matrix features = new Matrix();
			Matrix labels = new Matrix();
			features.setSize(rows, cols-1);
			labels.setSize(rows, 1);
			int l = 0;
			for (int i = 0; i < size; i++) {
				if(data.get(i)[index] == val) {
					int k = 0;
					for (int j = 0; j < cols; j++) {
						if(j == index)
							continue;
						features.set(l, k, data.get(i)[j]);
						k++;
					}
					labels.set(l, 0, targets.get(i)[0]);
					l++;
				}
			}
			child.setData(features, labels);
			children.add(child);
		}
		split.setChildren(children);		
	}
	
	public Map<Integer, Set<Double>> constructOutputs(Matrix features) {
		Map<Integer, Set<Double>> outputClasses = new HashMap<Integer, Set<Double>>();
		int size = features.m_enum_to_str.size();
		for (int i = 0; i < size; i++) {
			Set<Double> featureVals = new LinkedHashSet<Double>();
			int classSize = features.m_enum_to_str.get(i).size();
			for (int j = 0; j < classSize; j++) {
				featureVals.add((double)j);
			}
			outputClasses.put(i, featureVals);
		}
		return outputClasses;
	}
	
	public Map<Integer, Set<Double>> updateOutputClasses(int index, Map<Integer, Set<Double>> outClass) {
		Map<Integer, Set<Double>> outCopy = new HashMap<Integer, Set<Double>>();
		outCopy.putAll(outClass);
		int size = outClass.size();
		outCopy.remove(index);
		for (int i = index+1; i < size; i++) {
			Set<Double> tempSet = outCopy.get(i);
			outCopy.remove(i);
			outCopy.put(i-1, tempSet);
		}
		return outCopy;
	}

	public double findMostCommon(List<Double> outputs, Set<Double> vals) {
		double common = -1.0;
		int frequency = -1;
		for(Double val : vals) {
			int occur = Collections.frequency(outputs, val);
			if(occur > frequency) {
				frequency = occur;
				common = val;
			}
		}
		return common;
	}
	
	public double calculateGain(Node attr, Set<Double> outputs, List<Double> targets, double infoTotal) {
		List<Double> featureData = attr.getData();
		Set<Double> featureVals = attr.getVals();
		double size = featureData.size();
		double info = 0;
		layer.add(attr);
		for (Double val: featureVals) {
			ArrayList<Double> overlap = new ArrayList<Double>();
			int count = Collections.frequency(featureData, val);
			for (int i = 0; i < size; i++) {
				if(featureData.get(i).equals(val))
					overlap.add(targets.get(i));
			}
			info += (count / size) * calculateInfo(outputs, overlap);
		}
		attr.setInfo(info);
		return infoTotal - info;
		
	}
	
	public double calculateInfo(Set<Double> outputs, List<Double> targets) {
		double size = targets.size();
		double info = 0;
		double ratio = 0;
		// iterate over the possible outputs
		for (Double out : outputs) {
			// find the number of occurrences of the output
			int count = Collections.frequency(targets, out);
			if(size != 0)
				ratio = count / size;
			// calculate the info
			if(ratio != 0)
				info -= (ratio) * Math.log(ratio) / Math.log(2);
		}
		return info;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		int length = features.length;
		for(int i = 0; i < length; i++) {
			if(features[i] == Double.MAX_VALUE)
				features[i] = commonElement.get(i);
		}
		treePredict(features, labels, root);
		//System.out.format("%f %f %f %f %f\n", features[0], features[1], features[2], features[3], labels[0]);
	}
	
	public void treePredict(double[] features, double[] labels, Node currNode) throws Exception {
		if(currNode.getOutput() != -1) {
			labels[0] = currNode.getOutput();
		} else {
			int index = (int)currNode.getIndex();
			int value = (int)features[index];
			double[] newFeatures = new double[features.length - 1];
			int size = features.length;
			int k = 0;
			for (int i = 0; i < size; i++) {
				if (index == i)
					continue;
				newFeatures[k] = features[i];
				k++;
			}

			Node nextNode = currNode.getChildren().get(value);
			treePredict(newFeatures, labels, nextNode);
		}
	}
	
	public void printTree(Node n) {
		if(n.getChildren() == null) {
			System.out.format("Parent: %f Output: %f\n", n.getParent().getIndex(), n.getOutput());
			return;
		}
		else {
			if(n.getParent()==null)
				System.out.format("Index: %f Output: %f\n", n.getIndex(), n.getOutput());
			else
				System.out.format("Parent: %f Index: %f Output: %f\n", n.getParent().getIndex(), n.getIndex(), n.getOutput());
			for (Node child : n.getChildren()) {
				printTree(child);
			}
		}
	}

	@Override
	public double validate(Matrix features, Matrix labels) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}
}
