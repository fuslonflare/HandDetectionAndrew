import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_objdetect;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * Created by Phuwarin on 1/29/2017.
 */
public class Handy extends JFrame {
    // GUI components
    private HandPanel handPanel;

    public Handy() {
        super("Hand Detector");

        Container c = getContentPane();
        c.setLayout(new BorderLayout());

        // preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);

        handPanel = new HandPanel(); // the webcam pictures and drums appear here
        c.add(handPanel, BorderLayout.CENTER);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                handPanel.closeDown();
                System.exit(0);
            }
        });

        setResizable(false);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    } // end of defaultConstructor()

    public static void main(String[] args) {
        new Handy();
    } // end of main
} // end of Handy class
