interface StepProps {
  data: Record<string, unknown>
  onUpdate: (data: Record<string, unknown>) => void
}

export function Step5CookingSkill(_props: StepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Cooking Skill Level
        </h2>
        <p className="text-muted-foreground">
          How would you rate your cooking skills?
        </p>
      </div>

      <div className="space-y-4">
        {/* Add your form fields here */}
        <p className="text-center text-muted-foreground">
          Step 5 content - Cooking skill level selection will go here
        </p>
      </div>
    </div>
  )
}
