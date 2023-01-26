package loglocationrecommend.activator;

import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.IPath;
import org.eclipse.jdt.core.ICompilationUnit;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.internal.ui.javaeditor.CompilationUnitEditor;
import org.eclipse.jface.resource.ImageDescriptor;
import org.eclipse.jface.text.IDocument;
import org.eclipse.jface.text.TextSelection;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.ui.IEditorInput;
import org.eclipse.ui.ISelectionListener;
import org.eclipse.ui.ISelectionService;
import org.eclipse.ui.IWorkbench;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchPart;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.plugin.AbstractUIPlugin;
import org.eclipse.ui.texteditor.IDocumentProvider;
import org.osgi.framework.BundleContext;

import loglocationrecommend.view.ShowView;

public class Activator extends AbstractUIPlugin{
	private static Activator plugin;
	public static final String PLUGIN_ID = "loglocationrecommend.activator"; //$NON-NLS-1$
	public Activator() {
	}
	@Override
	public void start(BundleContext context) throws Exception {
		// TODO Auto-generated method stub
		plugin = this;
		super.start(context);
		IWorkbench workbench = PlatformUI.getWorkbench();

		IWorkbenchWindow activeWorkbenchWindow = workbench.getActiveWorkbenchWindow();

		ISelectionService selectionService = activeWorkbenchWindow.getSelectionService();
		final IWorkbenchPage activePage = activeWorkbenchWindow.getActivePage();
		selectionService.addSelectionListener(new ISelectionListener() {
			@Override
			public void selectionChanged(IWorkbenchPart part, ISelection selection) {
				if (part instanceof CompilationUnitEditor) {
					CompilationUnitEditor CUEditor = (CompilationUnitEditor) part;
					IDocumentProvider docProvider = CUEditor.getDocumentProvider();
					IEditorInput editorInput = CUEditor.getEditorInput();
					IFile file = (IFile) editorInput.getAdapter(IFile.class);
					IPath fullPath = file.getFullPath();
					ICompilationUnit workingCopy = getWorkingCopy(fullPath);
					IDocument document = docProvider.getDocument(editorInput);
					TextSelection tsel = (TextSelection) selection;
					//System.out.println(tsel.getStartLine()+" "+tsel.getEndLine());
					int pos = tsel.getOffset()+tsel.getLength();
					//activePage.showView("loglocationrecommend.view.ShowView");
					ShowView view = (ShowView) activePage.findView("loglocationrecommend.view.ShowView");
					IPath workspaceDir = ResourcesPlugin.getWorkspace().getRoot().getLocation();
					view.updateRes();
					CUEditor.setSelection(null);
					
					
					
				}
			}

			public ICompilationUnit getWorkingCopy(IPath path) {
				//get corresponding working copy
				ICompilationUnit compUnit = null;
				ICompilationUnit[] copies = JavaCore.getWorkingCopies(null);
				for (ICompilationUnit copy: copies){
					IPath path2 = copy.getPath().makeAbsolute();
					if (path.makeAbsolute().equals(path2)){
						compUnit = copy;
						break;
					}
				}
				return compUnit;
			}
		});
	}
	public void stop(BundleContext context) throws Exception {
		plugin = null;
		super.stop(context);
	}
	public static Activator getDefault() {
		return plugin;
	}
	public static ImageDescriptor getImageDescriptor(String path) {
		return imageDescriptorFromPlugin(PLUGIN_ID, path);
	}
}
