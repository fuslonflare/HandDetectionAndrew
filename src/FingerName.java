/**
 * Created by Phuwarin on 1/29/2017.
 */
public enum FingerName {
    LITTLE, RING, MIDDLE, INDEX, THUMB, UNKNOWN;

    public FingerName getNext() {
        int nextIdx = ordinal() + 1;
        if (nextIdx == (values().length)) {
            nextIdx = 0;
        }
        return values()[nextIdx];
    } // end of getNext()

    public FingerName getPrev() {
        int prevIdx = ordinal() - 1;
        if (prevIdx < 0) {
            prevIdx = values().length - 1;
        }
        return values()[prevIdx];
    } // end of getPrev()
} // end of FingerName enum
