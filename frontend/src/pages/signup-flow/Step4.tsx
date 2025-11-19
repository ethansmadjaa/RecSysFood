interface StepProps {
  data: Record<string, unknown>
  onUpdate: (data: Record<string, unknown>) => void
}

export function Step4CuisinePreferences(_props: StepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Cuisine Preferences
        </h2>
        <p className="text-muted-foreground">
          What types of cuisine do you enjoy?
        </p>
      </div>

      <div className="space-y-4">
        {/* Add your form fields here */}
        <p className="text-center text-muted-foreground">
          Step 4 content - Cuisine preferences selection will go here
        </p>
      </div>
    </div>
  )
}
