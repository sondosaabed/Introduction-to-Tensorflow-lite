import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/*
 *          This is the class that they are using to inference using TFLI?TE model 
 *          It is used to interpret & prepare the image for the preprocessing and for the model
 *             wE will be using something similer 
 *          Note that they used Kotlin but I copied the code and converted it to java since we are using Java
 */
public class Classifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private final int inputSize;
    private final int pixelSize = 3;
    private final int imageMean = 0;
    private final float imageStd = 255.0f;
    private final int maxResult = 3;
    private final float threshold = 0.4f;

    public Classifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize) throws IOException {
        this.inputSize = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelPath);
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        FileInputStream inputStream = assetManager.open(modelPath);
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileChannel.position();
        long declaredLength = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labels = new ArrayList<>();
        Scanner scanner = new Scanner(assetManager.open(labelPath));
        while (scanner.hasNextLine()) {
            labels.add(scanner.nextLine());
        }
        scanner.close();
        return labels;
    }

    public List<Recognition> recognizeImage(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        float[][] result = new float[1][labelList.size()];
        interpreter.run(byteBuffer, result);
        return getSortedResult(result);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * pixelSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                int input = intValues[pixel++];
                byteBuffer.putFloat(((input >> 16 & 0xFF) - imageMean) / imageStd);
                byteBuffer.putFloat(((input >> 8 & 0xFF) - imageMean) / imageStd);
                byteBuffer.putFloat(((input & 0xFF) - imageMean) / imageStd);
            }
        }
        return byteBuffer;
    }

    private List<Recognition> getSortedResult(float[][] labelProbArray) {
        Log.d("Classifier", "List Size: (" + labelProbArray.length + ", " + labelProbArray[0].length + ", " + labelList.size() + ")");
        PriorityQueue<Recognition> pq = new PriorityQueue<>(
                maxResult,
                new Comparator<Recognition>() {
                    @Override
                    public int compare(Recognition r1, Recognition r2) {
                        return Float.compare(r2.getConfidence(), r1.getConfidence());
                    }
                });

        for (int i = 0; i < labelList.size(); i++) {
            float confidence = labelProbArray[0][i];
            if (confidence >= threshold) {
                pq.add(new Recognition("" + i, (i < labelList.size()) ? labelList.get(i) : "Unknown", confidence));
            }
        }
        Log.d("Classifier", "pq size: (" + pq.size() + ")");

        List<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), maxResult);
        for (int i = 0; i < recognitionsSize; i++) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }
}