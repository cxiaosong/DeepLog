package loglocationrecommend.view;

import java.awt.Color;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.runtime.IPath;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jdt.core.ICompilationUnit;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.JavaModelException;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.internal.ui.javaeditor.CompilationUnitEditor;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.IRegion;
import org.eclipse.jface.text.ITextSelection;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.StackLayout;
import org.eclipse.swt.events.MouseAdapter;
import org.eclipse.swt.graphics.RGB;
import org.eclipse.swt.layout.FillLayout;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Display;
import org.eclipse.text.edits.DeleteEdit;
import org.eclipse.text.edits.InsertEdit;
import org.eclipse.text.edits.TextEdit;
import org.eclipse.ui.IEditorInput;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IWorkbench;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.part.ViewPart;
import org.eclipse.ui.texteditor.IDocumentProvider;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.ExpressionStmt;

import chang.predict.log.getAllMethod.ProcessModel1Data;
import loglocationrecommend.analysisAst.GetMethodMetrix;
import loglocationrecommend.analysisAst.GetPythonModelMessage;

/**
 * 
 * 
 * 
 *  @Description   : 展示重构结果界面（增加按钮监听）
 *  @Project       : LogLocationRecommend
 *  @Program Name  : loglocationrecommend.viewShowView.java
 *  @Author        : 常晓松
 *  @Creation Date : 2022年11月4日下午2:19:01
 *  @version       : v1.00
 */
public class ShowView extends ViewPart {
	private StackLayout stackLayout;
	private Composite superParent;
	private Composite videoPanel;
	private Frame videoFrame;
	private Button addLog ;
	private Button unLog ;
	public static String host="http://127.0.0.1:8003/test";
	static String logName="Logger"; 
 
	
	
	@Override
	public void createPartControl(Composite parent) {
		// TODO Auto-generated method stub
		superParent = parent;
		parent.setLayout(new FillLayout(SWT.HORIZONTAL));
//		Text text = new Text(topComp, SWT.BORDER);
//		text.setText("中国之编辑器");
        Button addlog=new Button(parent, SWT.NONE);
        
        addlog.setText("add-log");
        Button unlog=new Button(parent, SWT.NONE);
        unlog.setText("un-log");
		this.addLog=addlog;
		this.unLog=unlog;
		
		addLog.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseDown(org.eclipse.swt.events.MouseEvent e) {
				// TODO Auto-generated method stub
				super.mouseDown(e);

				IWorkbench workbench = PlatformUI.getWorkbench();
				IWorkbenchWindow activeWorkbenchWindow = workbench.getActiveWorkbenchWindow();
				IWorkbenchPage page = activeWorkbenchWindow.getActivePage();
				IEditorPart editor = page.getActiveEditor();
				CompilationUnitEditor CUEditor = (CompilationUnitEditor) editor;
				IDocumentProvider docProvider = CUEditor.getDocumentProvider();
				IEditorInput editorInput = CUEditor.getEditorInput();
				IFile file = (IFile) editorInput.getAdapter(IFile.class);
				
				IPath fullPath = file.getFullPath();
				ICompilationUnit workingCopy = getWorkingCopy(fullPath);
				IDocument document = docProvider.getDocument(editorInput);
				ISelection sel = editor.getEditorSite().getSelectionProvider()
						.getSelection();
				ITextSelection tSel = (ITextSelection) sel;
				
				
				int startLine=tSel.getStartLine();
				int endLine=tSel.getEndLine();

				int offset = tSel.getOffset();
				int length = tSel.getLength();
				int pos = offset + length;
				

				//判断是否有log
				String logParamNam=null;
				File newFile = new File(file.getLocation().toOSString());
				
				List<String> allLines =new ArrayList<String>();
				try {
					Path path = Paths.get(file.getLocation().toOSString());
					byte[] bytes = Files.readAllBytes(path);
					allLines = Files.readAllLines(path, StandardCharsets.UTF_8);
				} catch (IOException e2) {
					// TODO Auto-generated catch block
					e2.printStackTrace();
				}
				//更多请阅读：https://www.yiibai.com/java/java-read-text-file.html


			        
				CompilationUnit astAllMessage=null;
				try {
					int offSet=0;
					int alreadyInsertChar=0;
					astAllMessage = ProcessModel1Data.getCompilationUnit(newFile.getAbsolutePath());
					GetMethodMetrix getMethodMetrix = new GetMethodMetrix(startLine,endLine,true);
					getMethodMetrix.visit(astAllMessage, null);
					MethodDeclaration method=getMethodMetrix.getWriteMethod();
					//若有则读取名字
					logParamNam = getLoggerName(astAllMessage, method);
					
					//新增info语句
					insertLogInfo(workingCopy, document, tSel, pos, alreadyInsertChar);
					//若没有则新增声明
					if (logParamNam==null) {
						logParamNam = insertLogParameter(workingCopy, document, allLines, astAllMessage,
								alreadyInsertChar);
					}
					 
					
					
					//新增import
					boolean shouldAddImport=true;
					for (ImportDeclaration oneImport : astAllMessage.getImports()) {
						if (oneImport.getNameAsString().indexOf("Logger")>=0) {
							shouldAddImport=false;
						}
					}
					if (shouldAddImport) {
						workingCopy.createImport("org.apache.log4j.Logger", null, null);
					}
					
					
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				} catch (JavaModelException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				} catch (BadLocationException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
 
				
			}

			private void insertLogInfo(ICompilationUnit workingCopy, IDocument document, ITextSelection tSel, int pos,
					int alreadyInsertChar) throws BadLocationException, JavaModelException {
				//执行log语句的位置计算
				IRegion lineRegTmp;
				//执行插入操作
				lineRegTmp = document.getLineInformationOfOffset(pos);
				//执行插入
				String[] sentence=tSel.getText().split("\r\n");
				String  lastSentence=sentence[sentence.length-1];
				char[] charList=lastSentence.toCharArray();
				String retract="";
				for (char c : charList) {
					if (c=='\t') {
						retract=retract+"\t";
					}else {
						break;
					}
				}
				
				insertOneLine(workingCopy, document, lineRegTmp.getOffset()+alreadyInsertChar,retract+"log.info(\"autoLog\");");
			}

			private String insertLogParameter(ICompilationUnit workingCopy, IDocument document, List<String> allLines,
					CompilationUnit astAllMessage, int alreadyInsertChar)
					throws BadLocationException, JavaModelException {
				String logParamNam;
				int offSet;
				//暂时采用直接插入法，后续改善
				String className="";
				for (Node oneNode1 : astAllMessage.getChildNodes()) {
					if (oneNode1 instanceof ClassOrInterfaceDeclaration) {
						ClassOrInterfaceDeclaration ClassOrInterfaceDeclaration=(ClassOrInterfaceDeclaration)oneNode1;
						className=ClassOrInterfaceDeclaration.getName().toString();
						
					}
				}
				logParamNam="log";
				int offSet1=0;
				
				for (String oneSentence : allLines) {
					offSet1=offSet1+oneSentence.length()+2;
					if (oneSentence.indexOf(className)>=0) {
						offSet1=offSet1+oneSentence.length()+2;
						break;
					}
					
				}
				offSet=offSet1;
				
				insertOneLine(workingCopy, document, offSet+alreadyInsertChar,"\tprivate static Logger log = Logger.getLogger("+className+".class.getClass());");
				return logParamNam;
			}

			private int getSentencePoint(String[] classsContext, String shoudAddLocation) {
				int offSet=0;
				for (String oneSentence : classsContext) {
					offSet=offSet+oneSentence.length()+2;
					if (oneSentence.indexOf(shoudAddLocation)>=0) {
						break;
					}
					
				}
				return offSet;
			}

			private int insertOneLine(ICompilationUnit workingCopy, IDocument document, int pos,String sentence)
					throws BadLocationException, JavaModelException {
				IRegion lineReg;
				lineReg = document.getLineInformationOfOffset(pos);
				TextEdit edit = new InsertEdit(lineReg.getOffset(), sentence+"\r\n");
				workingCopy.applyTextEdit(edit, null);
				workingCopy.reconcile(ICompilationUnit.NO_AST, false, null, null);
				return sentence.length();
			}
		});
		
		unlog.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseDown(org.eclipse.swt.events.MouseEvent e) {
				// TODO Auto-generated method stub
				super.mouseDown(e);

				IWorkbench workbench = PlatformUI.getWorkbench();
				IWorkbenchWindow activeWorkbenchWindow = workbench.getActiveWorkbenchWindow();
				IWorkbenchPage page = activeWorkbenchWindow.getActivePage();
				IEditorPart editor = page.getActiveEditor();
				CompilationUnitEditor CUEditor = (CompilationUnitEditor) editor;
				IDocumentProvider docProvider = CUEditor.getDocumentProvider();
				IEditorInput editorInput = CUEditor.getEditorInput();
				IFile file = (IFile) editorInput.getAdapter(IFile.class);
				
				IPath fullPath = file.getFullPath();
				ICompilationUnit workingCopy = getWorkingCopy(fullPath);
				IDocument document = docProvider.getDocument(editorInput);
				ISelection sel = editor.getEditorSite().getSelectionProvider()
						.getSelection();
				ITextSelection tSel = (ITextSelection) sel;
				
				
				int startLine=tSel.getStartLine();
				int endLine=tSel.getEndLine();

				int offset = tSel.getOffset();
				int length = tSel.getLength();
				int pos = offset + length;
				IRegion lineReg;
				try {
					
					//获取每一条语句
					String[] sentence=tSel.getText().split(";");
					for (int i = sentence.length-1; i >=0 ; i--) {
						boolean isLogSentence=false;
						isLogSentence = checkIsLogSentence(isLogSentence,sentence[i]);
						if (isLogSentence) {
							TextEdit editDelete =new DeleteEdit(tSel.getText().indexOf(sentence[i])+1+offset, sentence[i].length());
							workingCopy.applyTextEdit(editDelete, null);
							workingCopy.reconcile(AST.JLS9, true, null, new NullProgressMonitor());
							
						}
					}
					
					//判断是否为log语句
					//若是则记录位置和长度

					//根据记录倒叙删除

					
					
					
					
					
					
					
					
					
					
					
					
					
					
					
					
				} catch (JavaModelException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
				
				
				
			}
		});
		
	} 
	private static boolean checkIsLogSentence(boolean isLogSentence,String sentence) {
		List<String> logMethodName=new ArrayList<String>();
		logMethodName.add("trace");
		logMethodName.add("debug");
		logMethodName.add("info");
		logMethodName.add("warn");
		logMethodName.add("error");
		logMethodName.add("warning");
		ParseResult<Expression> Expression=new JavaParser().parseExpression(sentence);
		
		if (!Expression.getResult().isPresent()) {
			return false;
		}
		Expression expressionResult= Expression.getResult().get();
		
		if (expressionResult instanceof MethodCallExpr) {
			
			if (logMethodName.contains(((MethodCallExpr) expressionResult).getName().toString())) {
				if ((((MethodCallExpr) expressionResult).getScope()).get().toString().indexOf("log")>=0) {
					isLogSentence=true;
				}
			}
		}
		return isLogSentence;
	}
	private void updateButtonValue(float rate) {
		// TODO Auto-generated method stub
		addLog.setText("add-log("+rate+")");
		unLog.setText("un-log("+(1-rate)+")");
		superParent.layout();
		superParent.redraw();
	
	}
	private float getPredictValue(String semanticMessageAll,String syntacticMessageAll) {
		// TODO Auto-generated method stub
		semanticMessageAll=GetPythonModelMessage.convertString(semanticMessageAll);
		syntacticMessageAll=GetPythonModelMessage.convertString(syntacticMessageAll);
		String rate=GetPythonModelMessage.doGet(host+"?syntatic="+syntacticMessageAll+"&"+"semantic="+semanticMessageAll);
		try {
			return Float.valueOf(rate.replaceAll("\"", ""));
			
		} catch (Exception e) {
			// TODO: handle exception
			return 0;
		}
	}
	public ICompilationUnit getWorkingCopy(IPath path) {
		// get corresponding working copy
		ICompilationUnit compUnit = null;
		ICompilationUnit[] copies = JavaCore.getWorkingCopies(null);
		for (ICompilationUnit copy : copies) {
			IPath path2 = copy.getPath().makeAbsolute();
			if (path.makeAbsolute().equals(path2)) {
				compUnit = copy;
				break;
			}
		}
		return compUnit;
	}
	public void updateRes() {
		// TODO Auto-generated method stub
		float shouldLogRate = getShouldLogRate();
		//根据预测值更新结果展示视图
		updateButtonValue(shouldLogRate);
		
		
		superParent.layout();
		superParent.redraw();
	}
	public void updateValue(float rate) {
		// TODO Auto-generated method stub
		if (addLog.getBackground().getRed()==255) {
			addLog.setBackground(
					new org.eclipse.swt.graphics.Color(Display.getDefault(), new RGB( 130, 227, 186 ) ));
			
		} else {
			addLog.setBackground(
					new org.eclipse.swt.graphics.Color(Display.getDefault(), new RGB(255,0,0 ) ));
		}
		superParent.layout();
		superParent.redraw();
	}
	
	@Override
	public void setFocus() {
		// TODO Auto-generated method stub

	}

	private float getShouldLogRate() {
		float shouldLogRate=0;
		//获取类信息
		IWorkbench workbench = PlatformUI.getWorkbench();
		IWorkbenchWindow activeWorkbenchWindow = workbench.getActiveWorkbenchWindow();
		IWorkbenchPage page = activeWorkbenchWindow.getActivePage();
		IEditorPart editor = page.getActiveEditor();
		CompilationUnitEditor CUEditor = (CompilationUnitEditor) editor;
		IDocumentProvider docProvider = CUEditor.getDocumentProvider();
		IEditorInput editorInput = CUEditor.getEditorInput();
		IFile file = (IFile) editorInput.getAdapter(IFile.class);
		
		IPath fullPath = file.getFullPath();
		ICompilationUnit workingCopy = getWorkingCopy(fullPath);
		IDocument document = docProvider.getDocument(editorInput);
		ISelection sel = editor.getEditorSite().getSelectionProvider()
				.getSelection();
		ITextSelection tSel = (ITextSelection) sel;
		
		
		int startLine=tSel.getStartLine();
		int endLine=tSel.getEndLine();

		
		
		//获取类的AST
		File newFile = new File(file.getLocation().toOSString());
		CompilationUnit astAllMessage=null;
		try {
			astAllMessage = ProcessModel1Data.getCompilationUnit(newFile.getAbsolutePath());
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			return shouldLogRate;
		}finally {
			if (astAllMessage==null) {
				return shouldLogRate;
			}
		}
		
		//根据AST确定目标方法
		//根据方法确定目标块
		GetMethodMetrix getMethodMetrix = new GetMethodMetrix(startLine,endLine);
		getMethodMetrix.visit(astAllMessage, null);
		//根据AST和目标块生成特征向量
		String semanticMessageAll=getMethodMetrix.getSemanticMessageAll();
		String syntacticMessageAll=getMethodMetrix.getSyntacticMessageAll();
		
		//调用python模块，传入向量，输出预测值
		shouldLogRate=getPredictValue(semanticMessageAll,syntacticMessageAll);
		return shouldLogRate;
	}
	
	public static void main(String[] args) {
		String file="D:\\workspace\\runtime-EclipseApplication\\test001\\src\\test001\\Test02.java";
		File newFile = new File(file);
		CompilationUnit astAllMessage=null;
		try {
			astAllMessage = ProcessModel1Data.getCompilationUnit(newFile.getAbsolutePath());
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		System.out.println(astAllMessage);
	}
	private static String getLoggerName(CompilationUnit astAllMessage,
			MethodDeclaration methodDeclaration) {
		 String logParamNam=null;
		 logParamNam=getMethodLoggerName(methodDeclaration);
		
		if (logParamNam==null) {
			logParamNam=getClassLoggerName(astAllMessage);
		}
		return logParamNam;
	}
	private static String getMethodLoggerName(MethodDeclaration methodDeclaration) {
		String logParamNam = null;
		for (Node oneNode  : methodDeclaration.getBody().get().getChildNodes()) {
			if (oneNode instanceof ExpressionStmt) {
				for (Node oneNode2  : oneNode.getChildNodes()) {
					if (oneNode2 instanceof  VariableDeclarationExpr) {
						for (Node oneNode3  : oneNode2.getChildNodes()) {
							if (oneNode3 instanceof 	VariableDeclarator ) {
								VariableDeclarator sentenceOne=(VariableDeclarator) oneNode3;
								if (logName.equals(sentenceOne.getType().toString())) {
									logParamNam=sentenceOne.getName().toString();
								}
							}
						}
					}
				}
			}
		}
		return logParamNam;
	}
	private static String getClassLoggerName(CompilationUnit astAllMessage) {
		String logParamNam = null;
		for (Node oneNodeTmp : astAllMessage.getChildNodes()) {
			if (oneNodeTmp instanceof ClassOrInterfaceDeclaration) {
				for(Node oneNode :oneNodeTmp.getChildNodes()) {
					if (oneNode instanceof FieldDeclaration) {
						FieldDeclaration variable=(FieldDeclaration)oneNode;
						if (logName.equals(variable.getCommonType().toString())) {
							for (Node littleNode : variable.getVariables().get(0).getChildNodes()) {
								if (littleNode instanceof SimpleName) {
									logParamNam =littleNode.toString();
									return logParamNam;
								}
							}
						}
						;
					}
					
				}
				
			}
		}
		return logParamNam;
	}
}
