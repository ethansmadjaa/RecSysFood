interface StepProps {
  data: Record<string, unknown>
  onUpdate: (data: Record<string, unknown>) => void
}

export function Step2DietaryPreferences(_props: StepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Dietary Preferences
        </h2>
        <p className="text-muted-foreground">
          Do you follow any specific diet?
        </p>
      </div>

      <div className="space-y-4">
        {/* Add your form fields here */}
        <p className="text-center text-muted-foreground">
          Step 2 content - Dietary preferences selection will go here
        </p>
      </div>
    </div>
  )
}
