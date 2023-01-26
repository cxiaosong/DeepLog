package loglocationrecommend.analysisAst;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.apache.log4j.Logger;

import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.visitor.GenericVisitorAdapter;

import chang.predict.log.parserMethodToBlock.ConvertBlockToTree;
import chang.predict.log.parserMethodToBlock.TreeNode;
import chang.predict.log.sumLogNumber.FilterAndSumLogBlock;
import chang.predict.log.vectorBlock.ProcessVectorMessage;
import loglocationrecommend.view.ShowView;
import lombok.Getter;
import lombok.Setter;

/**
 * 
 * 
 * 
 *  @Description   : 读取方法的度量信息 
 *  @Project       : LogLocationRecommend
 *  @Program Name  : loglocationrecommend.analysisAstMethodVisiter.java
 *  @Author        : 常晓松
 *  @Creation Date : 2022年11月4日下午2:21:03
 *  @version       : v1.00
 */

@Getter
@Setter
public class GetMethodMetrix  extends GenericVisitorAdapter<Void, Void>{
	 
	int startLine;
	int lineMedium;
	int endLine;
	boolean onlyGetMethod=false;
	String semanticMessageAll="";
	String syntacticMessageAll="";
	String filePathSemantic="D:\\python\\log_predict\\code_package\\getSemanticVec.py";
	String filePathSyntatic="D:\\python\\log_predict\\code_package\\getSyntaticMessage.py";
	
	MethodDeclaration writeMethod;
	
	public GetMethodMetrix() {
		// TODO Auto-generated constructor stub
		
	}
	
	public GetMethodMetrix(int startLine,
	int endLine) {
		// TODO Auto-generated constructor stub
		this.startLine=startLine;
		this.endLine=endLine;
	}
	public GetMethodMetrix(int startLine,
			int endLine,boolean onlyGetMethod) {
				// TODO Auto-generated constructor stub
				this.startLine=startLine;
				this.endLine=endLine;
				this.onlyGetMethod=onlyGetMethod;
			}
	@Override
	public Void visit(MethodDeclaration node, Void arg) { 
		int methodBeginLine=(Integer) node.getRange().map(new Function<Range, Object>() {
			public Object apply(Range r) {
				return r.begin.line;
			}
		}).orElse(-1);
		
		int methodStopLine=(Integer) node.getRange().map(new Function<Range, Object>() {
			public Object apply(Range r) {
				return r.end.line;
			}
		}).orElse(-1);
		
		lineMedium=(endLine+startLine+2)/2;
		
		//获取目标方法
		if (lineMedium>=methodBeginLine &&
				lineMedium<=methodStopLine) {
			//若仅获取当前光标所在的方法，则赋值，并返回
			if (onlyGetMethod) {
				this.writeMethod=node;
			}else {
				BlockStmt child = null;
				for (Node blockStmt : node.getChildNodes()) {
					if (blockStmt instanceof BlockStmt) {
						child = (BlockStmt) blockStmt;
					}
				}
				TreeNode root = new TreeNode(child, true, true,methodBeginLine,
						methodStopLine,0);
				
				
				
				
				//构建方法成树 
				ConvertBlockToTree.getChildBlock(root, 0);
				//获取目标块的向量表示
				List<TreeNode> chainToRoot=new ArrayList<TreeNode>();
				walkTree(chainToRoot,root,lineMedium);
				
				
				Logger log = Logger.getLogger(GetMethodMetrix.class.getClass());
				
				semanticMessageAll="";
				syntacticMessageAll="";
				for (TreeNode treeNode : chainToRoot) {
					String contex= treeNode.getContext();
					String contextChanged=FilterAndSumLogBlock.sumLogNumAndFilterLog(null, contex.split("\n"));
					String semanticMessage=ProcessVectorMessage.getSemiVec(contextChanged);
					String syntacticMessage=root.getSyntacticMessage();
					//生成向量信息
					semanticMessage=vecSemanticFromPython(semanticMessage);
					syntacticMessage=vecSyntaticFromPython(syntacticMessage);
					//返回向量信息
					semanticMessageAll=semanticMessageAll+semanticMessage.replaceAll("[\\t\\n\\r]", " ")
					.replaceAll("\"", "")
					.replaceAll(" +"," ").replaceAll(" ",",").replaceAll("\\\\n", "");
					syntacticMessageAll=syntacticMessageAll+syntacticMessage.replaceAll("[\\t\\n\\r]", " ")
					.replaceAll("\"", "")
					.replaceAll(" +"," ").replaceAll(" ",",").replaceAll("\\\\n", "");
				} 
				log.info(semanticMessageAll); 
				
			}
			
		}
		
		
		
		return super.visit(node, arg);
	
		 
		 
	}
	private String vecSemanticFromPython(String semanticMessage) {
		// TODO Auto-generated method stub
        String retValue=GetPythonModelMessage.doGet(ShowView.host+"?semantic="+GetPythonModelMessage.convertString(semanticMessage)+"");
        return retValue;
	 
	}
	private String vecSyntaticFromPython(String syntacticMessage) {
		// TODO Auto-generated method stub
        String retValue=GetPythonModelMessage.doGet(ShowView.host+"?syntatic="+GetPythonModelMessage.convertString(syntacticMessage)+"");
         
        return retValue;
	 
	}
	public static void walkTree(List<TreeNode> chainToRoot,TreeNode root,int lineMedium) {
		int fromLine=root.getFromLine();
		int toLine=root.getToLine();

		
		if (lineMedium>=fromLine &&
				lineMedium<=toLine) {
			chainToRoot.add(root);
		}
		
		//String 
		if (root.getChild()!=null) {
			for (TreeNode rootTmp : root.getChild()) {
				walkTree(chainToRoot,rootTmp,lineMedium);
			}
		}else {
			return;
		}
	}
	
}
