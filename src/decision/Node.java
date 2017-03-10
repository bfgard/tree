package decision;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class Node {

	private List<Double> data;
	private double info;
	private List<Node> children;
	private Set<Double> values;
	private Matrix features;
	private Matrix labels;
	private Node parent;
	private double output;
	private double index;
	
	public Node() {
		output = -1;
		index = -1;
		features = new Matrix();
	}
	
	public Node(Set<Double> vals, List<Double> data) {
		output = -1;
		index = -1;
		values = vals.stream().collect(Collectors.toSet());
		this.data = data.stream().collect(Collectors.toList());
	}
	
	public void setIndex(double index) {
		this.index = index;
	}
	
	public double getIndex() {
		return index;
	}
	
	public void setOut(double outputClass) {
		output = outputClass;
	}
	
	public double getOutput() {
		return output;
	}
	
	public void setData(Matrix data, Matrix targets) {
		features = data;
		labels = targets;
	}
	
	public Matrix getFeatures() {
		return features;
	}
	
	public Matrix getLabels() {
		return labels;
	}
	
	public Node getParent() {
		return parent;
	}
	
	public void setParent(Node n) {
		parent = n;
	}
	
	public void setChildren(List<Node> chldrn) {
		children = chldrn.stream().collect(Collectors.toList());;
	}
	
	public List<Node> getChildren() {
		return children;
	}
	
	public Set<Double> getVals() {
		return values;
	}
	
	public List<Double> getData() {
		return data;
	}
	
	public double getInfo() {
		return info;
	}
		
	public void setInfo(double info) {
		this.info = info;
	}
}
